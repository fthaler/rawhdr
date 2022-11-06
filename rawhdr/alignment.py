import cv2 as cv

from .common import reduce_color_dimension


def align(template, image, initial_transform=None, return_matrix=False):
    template_bw = reduce_color_dimension(template)
    image_bw = reduce_color_dimension(image)

    criteria = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 5000, 1e-6)
    ret, mat = cv.findTransformECC(template_bw, image_bw, initial_transform,
                                   cv.MOTION_AFFINE, criteria, None, 5)
    warped = cv.warpAffine(image,
                           mat,
                           image.shape[:2][::-1],
                           borderMode=cv.BORDER_REPLICATE,
                           flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)
    if return_matrix:
        return warped, mat
    return warped
