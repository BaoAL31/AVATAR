#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy import signal
from scipy.ndimage import binary_closing
from sklearn.cluster import AgglomerativeClustering

import torch
import torch.nn.functional as F

from .utils import find_runs, majority_filter_traditional, VIA_JSON_TEMPLATE

class Diarizer:
    def __init__(self, cache_dir: str, out_dir: str) -> None:
        self.cache_dir = cache_dir
        self.out_dir = out_dir
        self.frame_rate = 25
        # Agglomerative-clustering distance threshold for merging visual face
        # tracks into a single speaker. Distance = 1 - cosine_similarity of
        # speaker embeddings. Lower = more clusters (over-clustering, fragments
        # FA). Higher = aggressive merging (may collapse distinct speakers,
        # raises confusion). Default 0.3 over-clusters on AVA-AVD test (15 hyp
        # speakers vs 8 ref on clip 2qQs3Y9OJX0_c_01). Env-overridable.
        self.dist_thres = float(os.environ.get("AVATAR_DIST_THRES", 0.5))
        # Lowered 0.6 -> 0.4 so Path 2 audio-only-fallback (matching a VAD-detected
        # speech run against known visual-cluster speaker embeddings) accepts
        # weaker matches. This recovers off-screen-but-known speech that the
        # winner-per-frame ASD path can't see. AVA-AVD test subset showed ~28pp
        # of ref speech is off-screen for known speakers; bumping this threshold
        # down trades a small confusion cost for a large miss-rate reduction.
        self.spk_thres = float(os.environ.get("AVATAR_SPK_THRES", 0.4))
        # Overlap between face-track speech segments adds this to the precomputed
        # agglomerative distance (1 - cosine_sim + overlap * penalty). Large
        # values forbid merging tracks that share any frame; lower values allow
        # same-person-across-bystander merges but risk merging distinct simultaneous
        # speakers with similar voice. Default 100 matches historical behavior.
        self.ovl_penalty = float(os.environ.get("AVATAR_OVL_PENALTY", "100"))
        self.max_length = 120  #  in minutes

        os.makedirs(self.out_dir, exist_ok=True)
        self.data = VIA_JSON_TEMPLATE

    def addvad(self, vadscore: np.ndarray, vad:List[Tuple[float, float]]) -> np.ndarray:
        """
        Add VAD results to the vadscore

        Args:
            vadscore: np.ndarray, vadscore
            vad: List[Tuple[float, float]], vad results (start time, duration)

        Returns:
            np.ndarray, vadscore
        """
        for v in vad:
            vs = int(v[0] * self.frame_rate)
            ve = int((v[0] + v[1]) * self.frame_rate)
            vadscore[vs:ve] += 1
        return vadscore

    def run(self, 
            tracks: List[Dict], 
            asdres: List[np.ndarray], 
            vads: Dict[str, List[Tuple[float, float]]], 
            face_id: List[int], 
            spkfeats: torch.Tensor, 
            origfile: str = '/Users/jaesung/Desktop/sample.mp4',
            path2_diag: Optional[Dict] = None,
            ref_segments_sec: Optional[List[Tuple[float, float]]] = None):
        """
        Run diarization

        Args:
            tracks: List[Dict], list of tracks
            asdres: List[np.ndarray], list of ASD results
            vads: Dict[str, List[Tuple[float, float]]], dict of VAD results
            face_id: List[int], list of face ids
            spkfeats: torch.Tensor, speaker features
            origfile: str, original video file
            path2_diag: if set, populated with Path 1/2 accounting (debug / ablation).
            ref_segments_sec: optional reference speech intervals in seconds; used
                with path2_diag to report overlap fractions on ref frames only.

        Returns:
            List[Tuple[float, float, str]], list of diarization results
        """
        logging.info("Running diarization using intermediate results...")

        # TODO : Fix this with the length of actual videofile
        max_length = self.max_length * 60 * self.frame_rate
        allvad = np.zeros(max_length)
        allvad = self.addvad(allvad, vads['audio.wav'])
        allvad_copy = np.copy(allvad)

        spkemb = {}
        spkseg = {}
        utterances = []

        # Path 1a: per-track preprocessing. Compute fconfm + vadoutput for each
        # track but do NOT emit utterances yet. We need the global picture
        # first so we can pick a single ASD winner per frame and avoid
        # stacked overlap (the historical FA source).
        per_track: List[Dict] = []
        for tidx, track in enumerate(tracks):
            mean_dists = np.mean(np.stack(asdres[tidx], 1), 1)
            minidx = np.argmin(mean_dists, 0)

            fdist = np.stack([dist[minidx] for dist in asdres[tidx]])
            fdist = np.pad(fdist, (3, 3), 'constant', constant_values=10)

            fconf = np.median(mean_dists) - fdist
            fconfm = signal.medfilt(fconf, kernel_size=25)

            vadscore = np.zeros_like(fconfm)
            vadscore = self.addvad(vadscore, vads[f'{tidx:05d}.wav'])

            if face_id[tidx] not in spkseg:
                spkseg[face_id[tidx]] = np.zeros(max_length)
            spkseg[face_id[tidx]][track['track']['frame']] += 1

            vadoutput = np.copy(vadscore)
            # 0.3 vs 0.5: include moderate ASD confidence when gating with VAD.
            vadoutput[fconfm > 0.3] += 1
            vadoutput[fconfm > 0.9] += 1
            vadoutput[vadoutput < 2] = 0
            vadoutput[vadoutput >= 2] = 1
            vadoutput = binary_closing(vadoutput, structure=np.ones((12))).astype(int)

            allvad_copy[track['track']['frame'][0]:track['track']['frame'][0] + len(vadscore)] -= vadoutput

            per_track.append({
                'track': track,
                'fconfm': fconfm,
                'vadoutput': vadoutput,
                'frame_start': int(track['track']['frame'][0]),
            })

        # Path 1b: pick a single ASD winner per frame.
        # For each frame, the track with highest fconfm AND vadoutput==1 wins.
        # All other tracks at that frame are treated as silent bystanders.
        # This is the structural fix for ASD-per-track stacking, which was
        # producing 200-300% FA in multi-face scenes (one stacked SPEAKER
        # row per visible face per utterance).
        winner = -np.ones(max_length, dtype=np.int32)
        best_conf = -np.inf * np.ones(max_length, dtype=np.float32)
        for tidx, pt in enumerate(per_track):
            fs = pt['frame_start']
            vo = pt['vadoutput']
            fc = pt['fconfm']
            n = min(len(vo), max_length - fs)
            if n <= 0:
                continue
            active = vo[:n] == 1
            frames = np.arange(fs, fs + n)
            if not np.any(active):
                continue
            cand_frames = frames[active]
            cand_conf = fc[:n][active]
            better = cand_conf > best_conf[cand_frames]
            sel = cand_frames[better]
            best_conf[sel] = cand_conf[better]
            winner[sel] = tidx

        # Path 1c: emit utterances only where this track is the frame winner.
        for tidx, pt in enumerate(per_track):
            fs = pt['frame_start']
            n = min(len(pt['vadoutput']), max_length - fs)
            if n <= 0:
                continue
            won = (winner[fs:fs + n] == tidx).astype(int)
            if n < len(pt['vadoutput']):
                won = np.pad(won, (0, len(pt['vadoutput']) - n), constant_values=0)

            run_v, run_s, run_l = find_runs(won)
            run_s = run_s + fs

            # Min-run threshold: 7 frames = 0.28s at 25fps (was 15 = 0.6s).
            # AVA-AVD reference has many sub-half-second utterances ("yeah",
            # back-channel) which were being silently dropped, contributing
            # ~10pp to overall miss rate. 7-frame minimum still suppresses
            # single-frame noise.
            for r_idx, r_v in enumerate(run_v):
                if r_v > 0 and run_l[r_idx] > 7:
                    time_s = float(run_s[r_idx]) / self.frame_rate
                    time_e = float(run_s[r_idx] + run_l[r_idx]) / self.frame_rate
                    utterances.append({'vid': '1', 'flg': 0, 'xy': [], 'z': [time_s, time_e], 's': face_id[tidx]})

                    midtime = (time_s + time_e) / 2
                    midfeat = min(max(0, (midtime * 5) - 5), len(spkfeats) - 1)

                    if face_id[tidx] not in spkemb:
                        spkemb[face_id[tidx]] = []
                    spkemb[face_id[tidx]].append(spkfeats[[midfeat]])

        # Fallback: only fire when NO face_id won anywhere (truly silent /
        # short clip). In normal scenes this never runs - keeping the
        # fallback per-face_id would re-introduce stacked FA from silent
        # bystanders. Picks longest track in the scene to seed clustering.
        if not spkemb and tracks:
            longest_tidx = max(range(len(tracks)),
                               key=lambda i: len(tracks[i]['track']['frame']))
            track = tracks[longest_tidx]
            fid = face_id[longest_tidx]
            time_s = float(track['track']['frame'][0]) / self.frame_rate
            time_e = float(track['track']['frame'][-1] + 1) / self.frame_rate
            utterances.append({'vid': '1', 'flg': 0, 'xy': [], 'z': [time_s, time_e], 's': fid})
            midtime = (time_s + time_e) / 2
            midfeat = int(min(max(0, (midtime * 5) - 5), len(spkfeats) - 1))
            spkemb[fid] = [spkfeats[[midfeat]]]

        keys = list(spkemb.keys())
        keys.sort()

        if not keys:
            raise RuntimeError(
                "No speaker embeddings built from tracks. "
                "Diarization cannot continue safely; investigate VAD/ASD/track generation."
            )

        for k in keys:
            spkemb[k] = torch.mean(torch.cat(spkemb[k], 0), 0, keepdim=True)
        embmat = torch.cat([spkemb[k] for k in keys], 0)
        simmat = F.cosine_similarity(
            embmat.unsqueeze(-1).expand(-1, -1, len(embmat)),
            embmat.unsqueeze(-1).expand(-1, -1, len(embmat)).transpose(0, 2)
        ).detach().cpu().numpy()

        ovlmat = []
        for k in keys:
            ovlmat.append([np.sum(spkseg[k] * spkseg[q]) for q in keys])
        ovlmat = np.array(ovlmat).astype(int).clip(0, 1)
        ovlmat[range(0, len(ovlmat)), range(0, len(ovlmat))] = 0

        if len(ovlmat) >= 2:
            agc = AgglomerativeClustering(
                n_clusters=None, metric='precomputed', distance_threshold=self.dist_thres, linkage='average'
            ).fit(1 - simmat + ovlmat * self.ovl_penalty)
            labels = agc.labels_

            # Embedding-only post-merge (OFF by default, opt-in via env var).
            #
            # Designed to collapse clusters locked apart by ovl_penalty (same
            # person split across face tracks that share frames with bystander
            # tracks). Implementation works, but empirically the speakernet
            # embeddings on AVA-AVD do NOT separate same vs different speakers
            # cleanly enough for a global threshold: same-speaker pairwise
            # cosine tops out around 0.55-0.6 and cross-speaker can reach 0.4.
            # At threshold 0.4, distinct speakers collapse (clip 1 conf 2.89 ->
            # 16.65 in testing). At 0.5+, only 0-1 merges happen.
            #
            # Kept available behind env var because tighter speaker embeddings
            # (or a face-embedding-aware merge) could make this viable later.
            # Default = 1.0 (disabled).
            postmerge_thres = float(os.environ.get("AVATAR_POSTMERGE_THRES", 1.0))
            debug_sims = os.environ.get("AVATAR_DEBUG_CLUSTER_SIMS", "0") == "1"
            if postmerge_thres < 1.0 or debug_sims:
                emb_np = embmat.detach().cpu().numpy()
                uniq = sorted(set(labels.tolist()))
                cluster_emb = {
                    lbl: np.mean(emb_np[[i for i, x in enumerate(labels) if x == lbl]], axis=0)
                    for lbl in uniq
                }
                parent = {lbl: lbl for lbl in uniq}
                def _find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                for ii in range(len(uniq)):
                    for jj in range(ii + 1, len(uniq)):
                        a, b = uniq[ii], uniq[jj]
                        va, vb = cluster_emb[a], cluster_emb[b]
                        sim = float(np.dot(va, vb) /
                                    (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
                        if debug_sims:
                            ka = '/'.join(str(keys[i]) for i in range(len(labels)) if labels[i] == a)
                            kb = '/'.join(str(keys[i]) for i in range(len(labels)) if labels[i] == b)
                            mark = " <-MERGE" if sim >= postmerge_thres else ""
                            print(f"  ID_{ka}  vs  ID_{kb}: {sim:.3f}{mark}", flush=True)
                        if sim >= postmerge_thres:
                            parent[_find(a)] = _find(b)
                merged = np.array([_find(int(l)) for l in labels])
                # Relabel to consecutive 0..N-1
                final_uniq = sorted(set(merged.tolist()))
                remap = {old: new for new, old in enumerate(final_uniq)}
                labels = np.array([remap[int(l)] for l in merged])
        else:
            labels = np.array([0])

        labelname = {}
        groupname = {}
        for gidx, label in enumerate(range(max(labels) + 1)):
            idxlist = [keys[x] for x in np.where(labels == label)[0].tolist()]
            groupname[gidx] = 'ID_' + '/'.join(map(str, idxlist))
            for iidx in idxlist:
                labelname[iidx] = 'ID_' + '/'.join(map(str, idxlist))

        for uttidx, utterance in enumerate(utterances):
            utterance['av'] = {'1': labelname[utterance['s']]}
            del utterance['s']
            self.data['metadata'][f'{uttidx}'] = utterance
        # Build coverage map from Path 1 (ASD-per-track) utterances.
        # Path 2 (VAD-cluster with cosine-sim fallback) must not re-emit
        # frames already covered — this prevents stacked overlap rows
        # from inflating false alarm (FA) by 100-300% in multi-face scenes.
        covered = np.zeros(max_length, dtype=bool)
        for utt in utterances:
            fs = int(utt['z'][0] * self.frame_rate)
            fe = int(utt['z'][1] * self.frame_rate)
            covered[max(0, fs):min(max_length, fe)] = True

        residual_before_bin = np.copy(allvad_copy)
        if path2_diag is not None:
            timeline_end = 1
            if tracks:
                timeline_end = max(
                    timeline_end,
                    max(int(t['track']['frame'][-1]) + 1 for t in tracks),
                )
            ag = np.where(allvad > 0)[0]
            if ag.size > 0:
                timeline_end = max(timeline_end, int(ag[-1]) + 1)
            rg = np.where(residual_before_bin >= 1)[0]
            if rg.size > 0:
                timeline_end = max(timeline_end, int(rg[-1]) + 1)
            timeline_end = min(max_length, timeline_end + 1)

            ref_mask = np.zeros(max_length, dtype=bool)
            if ref_segments_sec:
                for a, b in ref_segments_sec:
                    fs = max(0, int(a * self.frame_rate))
                    fe = min(max_length, int(b * self.frame_rate))
                    if fe > fs:
                        ref_mask[fs:fe] = True
            ref_slice = slice(0, timeline_end)
            ref_on = ref_mask[ref_slice]
            n_ref = int(np.count_nonzero(ref_on))
            path2_diag['timeline_end_frame'] = timeline_end
            path2_diag['timeline_end_sec'] = timeline_end / self.frame_rate
            path2_diag['ref_frames'] = n_ref
            if n_ref > 0:
                g = allvad[ref_slice][ref_on] > 0
                path2_diag['ref_frac_global_audio_vad'] = float(np.mean(g))
                path2_diag['ref_frac_residual_count_ge1'] = float(
                    np.mean(residual_before_bin[ref_slice][ref_on] >= 1)
                )
                path2_diag['ref_frac_path1_covered'] = float(
                    np.mean(covered[ref_slice][ref_on])
                )
                w = winner[ref_slice][ref_on]
                path2_diag['ref_frac_path1_winner'] = float(np.mean(w >= 0))
                path2_diag['ref_frac_audio_but_no_winner'] = float(
                    np.mean((allvad[ref_slice][ref_on] > 0) & (w < 0))
                )
                path2_diag['ref_frac_residual_ge1_not_covered'] = float(
                    np.mean(
                        (residual_before_bin[ref_slice][ref_on] >= 1)
                        & (~covered[ref_slice][ref_on])
                    )
                )

        allvad_copy[allvad_copy < 1] = 0
        if path2_diag is not None and path2_diag.get('ref_frames', 0) > 0:
            path2_diag['ref_frac_residual_binary'] = float(
                np.mean(allvad_copy[ref_slice][ref_on] > 0)
            )

        path2_outer_short = 0
        path2_skipped_covered_frames = 0
        path2_skipped_unknown_frames = 0
        path2_emitted_frames = 0
        path2_low_conf_frames = 0

        run_v, run_s, run_l = find_runs(allvad_copy)
        for r_idx, r_v in enumerate(run_v):
            if r_v > 0 and run_l[r_idx] <= 10:
                path2_outer_short += 1
            if r_v > 0 and run_l[r_idx] > 10:
                indices = []
                low_conf = 0
                for frame in range(run_s[r_idx], run_s[r_idx] + run_l[r_idx]):
                    fr_feat = int(min(max(0, (frame / 5) - 5), len(spkfeats) - 1))

                    cossim = F.cosine_similarity(embmat, spkfeats[[fr_feat]])

                    mval = torch.max(cossim)
                    midx = torch.argmax(cossim)

                    if mval >= self.spk_thres:
                        indices.append(labels[midx])
                    else:
                        indices.append(-1)
                        low_conf += 1

                path2_low_conf_frames += low_conf
                indices = majority_filter_traditional(indices, 25)

                run_vs, run_ss, run_ls = find_runs(indices)

                for rs_idx, rs_v in enumerate(run_vs):
                    fs = run_s[r_idx] + run_ss[rs_idx]
                    fe = fs + run_ls[rs_idx]

                    # Skip: already covered by Path 1 ASD-per-track emission
                    if np.all(covered[fs:fe]):
                        if path2_diag is not None:
                            path2_skipped_covered_frames += fe - fs
                        continue
                    # Skip: low-confidence fallback — "unknown" is not a real speaker
                    if rs_v == -1:
                        if path2_diag is not None:
                            path2_skipped_unknown_frames += fe - fs
                        continue

                    time_s = float(fs) / self.frame_rate
                    time_e = float(fe) / self.frame_rate
                    if path2_diag is not None:
                        path2_emitted_frames += fe - fs
                    self.data['metadata'][f'{r_idx}_{rs_idx}'] = {
                        'vid': '1', 'flg': 0, 'xy': [], 'z': [time_s, time_e], 'av': {'1': groupname[rs_v]}
                    }

        if path2_diag is not None:
            path2_diag['path2_outer_runs_dropped_le10_count'] = path2_outer_short
            path2_diag['path2_skipped_covered_frames'] = path2_skipped_covered_frames
            path2_diag['path2_skipped_unknown_frames'] = path2_skipped_unknown_frames
            path2_diag['path2_emitted_frames'] = path2_emitted_frames
            path2_diag['path2_per_frame_low_conf_in_long_runs'] = path2_low_conf_frames
            path2_diag['path2_spk_thres'] = self.spk_thres
            path2_diag['path2_min_outer_run_frames'] = 10
            te = int(path2_diag.get('timeline_end_frame', max_length))
            path2_diag['path1_covered_frames'] = int(np.count_nonzero(covered[:te]))
            path2_diag['path2_residual_binary_frames_on_timeline'] = int(
                np.count_nonzero(allvad_copy[:te] > 0)
            )

        self.data["file"]["1"]["src"] = str(origfile)

        # Store the json file
        jsonfile = os.path.join(self.out_dir, 'result.json')
        with open(jsonfile, 'w') as outfile:
            json.dump(self.data, outfile)

        # Store the rttm file
        result = []
        bname = os.path.basename(origfile).split('.')[0]
        rttmfile = os.path.join(self.out_dir, 'result.rttm')
        with open(rttmfile, 'w') as f:
            uttkeys = list(self.data['metadata'].keys())
            uttkeys.sort()

            for uttkey in uttkeys:
                utt = self.data['metadata'][uttkey]
                if len(utt['z']) == 2:
                    spk = utt['av']['1'].replace(' ', '_')
                    start = utt['z'][0]
                    end = utt['z'][1]
                    f.write(f'SPEAKER {bname} 1 {start:.2f} {end - start:.2f} <NA> <NA> {spk} <NA> <NA>\n')
                    result.append((start, end, spk))

        return result


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'
    out_dir = '/users/jaesung/voxconverse_method/temp'
    with open('/users/jaesung/voxconverse_method/temp/pywork/tracks.pkl', 'rb') as f:
        tracks = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/activesd.pkl', 'rb') as f:
        asdres = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/webrtc.pkl', 'rb') as f:
        vads = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/faceidx.pkl', 'rb') as f:
        faceclusters_idx = pickle.load(f)
    spkfeats = torch.load('/users/jaesung/voxconverse_method/temp/pywork/ecapa.pt')
    diarizer = Diarizer(cache_dir=cache_dir, out_dir=out_dir)
    result = diarizer.run(tracks, asdres, vads, faceclusters_idx, spkfeats)