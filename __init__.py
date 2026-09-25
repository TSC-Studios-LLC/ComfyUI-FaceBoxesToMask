"""Turn the app's face-detection JSON into a mask and a crop box.

The FakeMe app already knows where every face is: `ImagePickerView` runs MLKit
with tracking over the clip and `refactorFaceOutput` regroups the tracking ids
into stable identities, saved as

    {"faceRectangles": [{"frame": 0, "rects": [{"id": 3, "x": .., "y": ..,
                                               "width": .., "height": ..}]}, ...],
     "faceIds": [3, 7]}

`frame` indexes the source's own frames; rectangles are pixels in the source
resolution, origin top-left. Nothing in ComfyUI accepts that shape, which is
why it otherwise has to be rendered into a mask clip and uploaded as a video.
This node takes it as a string instead.

Two outputs, from one parse:

  MASK   one frame per generated frame, white where that face is. Intersect it
         with SAM3's head mask (MaskComposite, multiply) to confine a swap to
         one person while keeping SAM3's hairline.
  x/y/w/h  the envelope of that face over the whole clip, for ImageCrop, so a
         pass only ever sees the person it is about.

Install: drop this folder in ComfyUI/custom_nodes/ and restart.
"""

import json

import torch

CATEGORY = "mask/faces"


def _parse(payload):
    """The app's JSON, however it arrives: bare, or under detected_faces."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    if "faceRectangles" not in data:
        for key in ("detected_faces", "detectedFaces"):
            if isinstance(data.get(key), dict):
                data = data[key]
                break
    if "faceRectangles" not in data:
        raise ValueError("no faceRectangles in the detection JSON")
    boxes = {}
    for entry in data["faceRectangles"]:
        frame = int(entry.get("frame", 0))
        for rect in entry.get("rects", []):
            boxes.setdefault(int(rect["id"]), {})[frame] = (
                float(rect["x"]), float(rect["y"]),
                float(rect["width"]), float(rect["height"]))
    ids = [int(i) for i in data.get("faceIds", sorted(boxes))]
    return ids, boxes


def _grown(box, grow_up, grow_side, grow_down, width, height):
    """MLKit boxes the face; a head swap needs the head, and the hair the box
    misses is above it — so the growth is asymmetric by default."""
    x, y, w, h = box
    return (max(0.0, x - w * grow_side),
            max(0.0, y - h * grow_up),
            min(float(width), x + w * (1.0 + grow_side)),
            min(float(height), y + h * (1.0 + grow_down)))


def _box_at(boxes, frame, hold):
    """That face's rectangle on a frame, carrying the last known one over a
    short detection dropout so the mask does not blink off."""
    if frame in boxes:
        return boxes[frame]
    if hold > 0:
        prior = [f for f in boxes if 0 <= frame - f <= hold]
        if prior:
            return boxes[max(prior)]
    return None


class FaceBoxesToMask:
    """The app's detection JSON -> a mask batch for one face."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The frames being generated; "
                                                "sets the mask's size and count."}),
                "detection_json": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "The app's detected.json"}),
                "face_id": ("INT", {"default": 0, "min": -1, "max": 4096,
                                    "tooltip": "Which faceId to mask. -1 masks every "
                                               "detected face."}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                         "step": 0.01,
                                         "tooltip": "Frame rate the JSON's frame "
                                                    "indices are in. The clip is "
                                                    "loaded at force_rate, so this is "
                                                    "the ORIGINAL rate."}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                         "step": 0.01}),
                "frame_offset": ("INT", {"default": 0, "min": 0, "max": 100000,
                                         "tooltip": "First generated frame's index in "
                                                    "the clip, for windowed runs."}),
                "grow_up": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0,
                                      "step": 0.05}),
                "grow_side": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 3.0,
                                        "step": 0.05}),
                "grow_down": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 3.0,
                                        "step": 0.05}),
                "hold": ("INT", {"default": 6, "min": 0, "max": 120}),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "x", "y", "width", "height", "frames_masked")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def build(self, images, detection_json, face_id, source_fps, target_fps,
              frame_offset, grow_up, grow_side, grow_down, hold):
        count, height, width = images.shape[0], images.shape[1], images.shape[2]
        ids, boxes = _parse(detection_json)
        wanted = ids if face_id < 0 else [face_id]
        missing = [i for i in wanted if i not in boxes]
        if missing:
            raise ValueError(f"face {missing} is not in the JSON; it has {ids}")

        mask = torch.zeros((count, height, width), dtype=torch.float32)
        env = [float("inf"), float("inf"), 0.0, 0.0]
        ratio = source_fps / target_fps
        painted = 0
        for t in range(count):
            source_frame = int(round((t + frame_offset) * ratio))
            hit = False
            for fid in wanted:
                box = _box_at(boxes[fid], source_frame, hold)
                if box is None:
                    continue
                x0, y0, x1, y1 = _grown(box, grow_up, grow_side, grow_down,
                                        width, height)
                ix0, iy0 = int(round(x0)), int(round(y0))
                ix1, iy1 = int(round(x1)), int(round(y1))
                if ix1 > ix0 and iy1 > iy0:
                    mask[t, iy0:iy1, ix0:ix1] = 1.0
                    env = [min(env[0], x0), min(env[1], y0),
                           max(env[2], x1), max(env[3], y1)]
                    hit = True
            painted += 1 if hit else 0

        if painted == 0:
            raise ValueError(
                f"face {wanted} has no rectangles in frames "
                f"{frame_offset}..{frame_offset + count} at {source_fps}fps")

        # Snap the crop box out to a multiple of 8 so ImageCrop lands on clean
        # pixels and the later composite has no half-pixel offset.
        x0 = max(0, int(env[0]) // 8 * 8)
        y0 = max(0, int(env[1]) // 8 * 8)
        x1 = min(width, -(-int(env[2]) // 8) * 8)
        y1 = min(height, -(-int(env[3]) // 8) * 8)
        return (mask, x0, y0, x1 - x0, y1 - y0, painted)


class FaceBoxesOverlap:
    """Do two faces' envelopes overlap?

    A crop that holds one head needs no mask: SAM3 finds the only head in it.
    A crop that holds two does, or the pass swaps both. This reports the
    overlap so a graph (or the caller) can pick.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detection_json": ("STRING", {"multiline": True, "default": ""}),
                "face_id": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "other_face_id": ("INT", {"default": 1, "min": 0, "max": 4096}),
                "grow_up": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0,
                                      "step": 0.05}),
                "grow_side": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 3.0,
                                        "step": 0.05}),
                "grow_down": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 3.0,
                                        "step": 0.05}),
                "tolerance": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0,
                                        "step": 0.05,
                                        "tooltip": "How much the envelopes may "
                                                   "overlap before it counts, as a "
                                                   "fraction of the other face's "
                                                   "typical box. The grown boxes carry "
                                                   "hair margin, so touching edges are "
                                                   "not two heads in one crop."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("overlap_x", "overlap_y", "overlaps")
    FUNCTION = "check"
    CATEGORY = CATEGORY

    def check(self, detection_json, face_id, other_face_id,
              grow_up, grow_side, grow_down, tolerance=0.25):
        _, boxes = _parse(detection_json)
        envs = {}
        for fid in (face_id, other_face_id):
            if fid not in boxes:
                raise ValueError(f"face {fid} is not in the JSON")
            env = [float("inf"), float("inf"), 0.0, 0.0]
            for box in boxes[fid].values():
                x0, y0, x1, y1 = _grown(box, grow_up, grow_side, grow_down,
                                        10 ** 6, 10 ** 6)
                env = [min(env[0], x0), min(env[1], y0),
                       max(env[2], x1), max(env[3], y1)]
            envs[fid] = env
        a, b = envs[face_id], envs[other_face_id]
        ox = int(min(a[2], b[2]) - max(a[0], b[0]))
        oy = int(min(a[3], b[3]) - max(a[1], b[1]))
        other = boxes[other_face_id].values()
        typical = sorted(w for _, _, w, _ in other)[len(list(other)) // 2] \
            if other else 0.0
        slack = typical * tolerance
        return (ox, oy, ox > slack and oy > slack)


def _overlap(mask, rect):
    """How much of a face's rectangle a tracked object covers.

    Scored this way round on purpose. IoU punishes the right object for being
    bigger than the box, and so does the share of the object that lies inside
    it: SAM3's head mask carries the whole length of the hair, so for the
    correct person only a fraction of it falls within a face-sized rectangle —
    measured at 0.19 on a three-face clip, which is indistinguishable from a
    miss. How much of the rectangle is covered does not care how much hair the
    mask has: it is near 1 for the person whose face that is, and near 0 for
    everyone else.
    """
    x0, y0, x1, y1 = rect
    area = (x1 - x0) * (y1 - y0)
    if area <= 0 or float(mask.sum()) <= 0:
        return 0.0
    return float(mask[y0:y1, x0:x1].sum()) / float(area)


class FaceBoxesPickObject:
    """Which SAM3 object is this face?

    SAM3 numbers the objects it tracked in its own order, and nothing ties
    object 0 to the app's face 0 — which is why a mask has to be intersected
    with a rectangle to pick a person. Matching the rectangle against the
    tracked objects instead gives the index, and then SAM3's own mask can be
    used unaltered: an organic hairline rather than a box with square corners.

    Feed the returned string to SAM3_TrackToMask's object_indices.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "detection_json": ("STRING", {"multiline": True, "default": ""}),
                "face_id": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                         "step": 0.01}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                         "step": 0.01}),
                "frame_offset": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "samples": ("INT", {"default": 8, "min": 1, "max": 64,
                                    "tooltip": "Frames to score, taken from "
                                               "the ones where this face is "
                                               "present. A handful is plenty "
                                               "and keeps this cheap."}),
                "min_overlap": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0,
                                          "step": 0.05,
                                          "tooltip": "Share of the face's "
                                                     "rectangle the object must "
                                                     "cover. Below this the "
                                                     "match is not believed and "
                                                     "the node raises rather "
                                                     "than swap the wrong "
                                                     "person."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("object_indices", "object_index", "overlap")
    FUNCTION = "pick"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def pick(self, track_data, detection_json, face_id, source_fps, target_fps,
             frame_offset, samples, min_overlap):
        from comfy.ldm.sam3.tracker import unpack_masks

        _, boxes = _parse(detection_json)
        if face_id not in boxes:
            raise ValueError(f"face {face_id} is not in the detection JSON")
        packed = track_data["packed_masks"]
        height, width = track_data["orig_size"]
        if packed is None:
            raise ValueError("SAM3 tracked nothing in this clip")
        frames, objects = packed.shape[0], packed.shape[1]

        # Score only on frames where this face is actually in shot. Walking
        # the clip at a fixed stride instead would hand a face that appears
        # halfway through — say, someone entering at seven seconds — mostly
        # frames it is absent from, and an object scored where the person is
        # not there is noise at best and a wrong match at worst.
        ratio = source_fps / target_fps
        present = []
        for native in sorted(boxes[face_id]):
            t = int(round(native / ratio)) - frame_offset
            if 0 <= t < frames:
                present.append((t, boxes[face_id][native]))
        if not present:
            raise ValueError(f"face {face_id} has no rectangles in this window")
        # Spread the samples over that face's own span.
        step = max(1, len(present) // samples)
        chosen = present[::step][:samples]

        scores = [0.0] * objects
        counted = len(chosen)
        for index in range(objects):
            masks = unpack_masks(packed[:, index])
            # The packed masks are at the tracker's own resolution; the app's
            # rectangles are in source pixels.
            scale_y = masks.shape[1] / float(height)
            scale_x = masks.shape[2] / float(width)
            for t, (x, y, w, h) in chosen:
                rect = (max(0, int(x * scale_x)), max(0, int(y * scale_y)),
                        min(masks.shape[2], int((x + w) * scale_x)),
                        min(masks.shape[1], int((y + h) * scale_y)))
                scores[index] += _overlap(masks[t], rect)
        best = max(range(objects), key=lambda i: scores[i])
        overlap = scores[best] / counted
        if overlap < min_overlap:
            raise ValueError(
                f"no SAM3 object matches face {face_id} (best overlap "
                f"{overlap:.2f} < {min_overlap:.2f}); the rectangles and the "
                "clip may not be the same resolution")
        return (str(best), best, overlap)


NODE_CLASS_MAPPINGS = {
    "FaceBoxesToMask": FaceBoxesToMask,
    "FaceBoxesOverlap": FaceBoxesOverlap,
    "FaceBoxesPickObject": FaceBoxesPickObject,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceBoxesToMask": "Face Boxes to Mask",
    "FaceBoxesOverlap": "Face Boxes Overlap",
    "FaceBoxesPickObject": "Face Boxes Pick SAM3 Object",
}
