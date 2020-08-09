from math import ceil

__all__ = ["insert_image_clipped"]


def insert_image_clipped(
    pos, original, inserting_im, inplace=False, name="", alpha_thresh=0
):
    """
    Insert one tensor into another dealing with areas that can't be inserted.
    """

    im = original if inplace else original.clone()
    im = im.permute(1, 2, 0)
    inserting_im = inserting_im.permute(1, 2, 0)
    cshape = im.shape
    imshape = inserting_im.shape
    pos = [*pos[::-1]]
    pos[0] = int(pos[0])
    pos[1] = int(pos[1])

    top_lx = pos[0] - ceil(imshape[0] / 2)
    top_ly = pos[1] - ceil(imshape[1] / 2)

    start_x1 = max(0, top_lx)
    end_x1 = min(pos[0] + int(imshape[0] / 2), cshape[0])

    start_y1 = max(0, top_ly)
    end_y1 = min(pos[1] + int(imshape[1] / 2), cshape[1])

    sx2 = 0
    if top_lx < 0:
        sx2 = abs(top_lx)
    ex2 = min(cshape[0] - top_lx, imshape[0])

    sy2 = 0
    if top_ly < 0:
        sy2 = abs(top_ly)
    ey2 = min(cshape[1] - top_ly, imshape[1])

    if ey2 - sy2 < 0 or ex2 - sx2 < 0:
        warnings.warn(f"{name} {original.shape} doesn't fit in {inserting_im.shape}")

    paste = im[start_x1:end_x1, start_y1:end_y1, :]
    if inserting_im.shape[-1] == 4:
        m = inserting_im[sx2:ex2, sy2:ey2, 3] > alpha_thresh
        paste[m] = inserting_im[sx2:ex2, sy2:ey2, :][m]
    paste = inserting_im[sx2:ex2, sy2:ey2, :]
    return im.permute(2, 0, 1)
