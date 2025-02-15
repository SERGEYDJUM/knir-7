from dataclasses import dataclass


class LesionBBox:
    def __init__(
        self,
        center: tuple[int, int] | tuple[int, int, int],
        r: int | tuple[int, int] | tuple[int, int, int],
        safe_r: int | tuple[int, int] | tuple[int, int, int] = None,
        roi_r: int | tuple[int, int] | tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        self.center = center
        self.r = r
        self.sr = safe_r
        self.rr = roi_r

        if isinstance(r, int):
            self.r = (r, r, r)

        if isinstance(safe_r, int):
            self.sr = (safe_r, safe_r, safe_r)

        if isinstance(roi_r, int):
            self.rr = (roi_r, roi_r, roi_r)

    def _bbox_2d(self, r: tuple) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            self.center[0] - r[0],
            self.center[1] - r[1],
        ), (
            self.center[0] + r[0],
            self.center[1] + r[1],
        )

    def _bbox_3d(self, r: tuple) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (
            self.center[0] - r[0],
            self.center[1] - r[1],
            self.center[2] - r[2],
        ), (
            self.center[0] + r[0],
            self.center[1] + r[1],
            self.center[2] + r[2],
        )

    def bbox(self, include_z: bool = False) -> tuple[tuple, tuple]:
        return self._bbox_3d(self.r) if include_z else self._bbox_2d(self.r)

    def safe_bbox(self, include_z: bool = False) -> tuple[tuple, tuple]:
        return self._bbox_3d(self.sr) if include_z else self._bbox_2d(self.sr)

    def roi_bbox(self, include_z: bool = False) -> tuple[tuple, tuple]:
        return self._bbox_3d(self.rr) if include_z else self._bbox_2d(self.rr)

    def scale(self, scale_factor: float = 1.0, imcenter=(256, 256, 150)):
        def scaler(x: int) -> int:
            return int(x * scale_factor)

        def retuple(t: tuple) -> tuple:
            return tuple(map(scaler, t))

        def recenter(t: tuple) -> tuple:
            shifted = map(lambda p: p[1] - p[0], zip(imcenter, self.center))
            scaled = retuple(tuple(shifted))
            return tuple(map(lambda p: p[1] + p[0], zip(imcenter, scaled)))

        return LesionBBox(
            center=recenter(self.center),
            r=retuple(self.r),
            safe_r=retuple(self.sr),
            roi_r=retuple(self.rr),
        )


@dataclass
class Phantom:
    signals: list[LesionBBox]
