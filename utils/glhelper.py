# coding: UTF-8
import numpy as np
import scipy.interpolate as inter
from PIL import ImageFont, ImageDraw, Image
from io import BytesIO
import base64
import pygltflib


fl32 = np.float32
font = ImageFont.truetype("arial.ttf", 200)

def sanitize_color(col):
    col = tuple(col)
    if len(col) == 4:
        return col
    elif len(col) == 3:
        return (*col, 1.)
    elif len(col) == 2:
        return (col[0], col[0], col[0], col[1])
    elif len(col) == 1:
        return (col[0], col[0], col[0], 1.)


def sanitize_primitive(points, norms=None, inds=None, uv=None):
    if inds is None:
        inds = np.arange(points.shape[0])
    else:
        inds = inds.flatten()

    triangles = inds[:inds.shape[0] // 3 * 3].reshape((-1, 3))
    triangles_nonan = triangles[~np.any(np.isnan(points[triangles]), axis=(1, 2))]

    inds_all = triangles_nonan.flatten()

    inds_unique, inds_new = np.unique(inds_all, return_inverse=True)
    points_new = points[inds_unique]
    if norms is not None:
        norms_new = norms[inds_unique]
    else:
        norms_new = None

    if uv is not None:
        uv_new = uv[inds_unique]
    else:
        uv_new = None
    return points_new, norms_new, inds_new, uv_new


def gen_material(color, transparent=False, doubleSided=False, emissive=False, metallicFactor=0., roughnessFactor=1., texIndex=None):
    color = sanitize_color(color)
    if texIndex is not None:
        texinfo = pygltflib.TextureInfo(index=texIndex)
    else:
        texinfo = None
    pbr = pygltflib.PbrMetallicRoughness(metallicFactor=metallicFactor, roughnessFactor=roughnessFactor, baseColorTexture=texinfo)
    material = pygltflib.Material()
    if emissive:
        pbr.baseColorFactor = (0., 0., 0., color[3])
        material.emissiveFactor = color[:3]
    else:
        pbr.baseColorFactor = color

    material.pbrMetallicRoughness = pbr
    material.doubleSided = doubleSided
    if transparent:
        material.alphaMode = pygltflib.BLEND
    else:
        material.alphaMode = pygltflib.MASK
    return material


def gen_image(array):
    im = Image.fromarray(array)
    io = BytesIO()
    im.save(io, format="png")
    io.seek(0)

    uri = f"data:image/png;base64,{base64.b64encode(io.read()).decode('utf-8')}"
    return pygltflib.Image(uri=uri)

def gen_text_image(text, color=(255, 255, 255)):
    text_lines = text.split("\n")
    ascent, descent = font.getmetrics()

    line_heights = [ascent + descent] * len(text_lines)

    total_height = sum(line_heights)
    total_width = max([font.getmask(line).getbbox()[2] for line in text_lines])

    img = Image.new("L", (total_width, total_height), color=(0,))
    draw_interface = ImageDraw.Draw(img)

    y = 0
    for i, line in enumerate(text_lines):
        line_width = font.getmask(line).getbbox()[2]
        x = ((total_width - line_width) // 2)

        draw_interface.text((x, y), line, font=font, fill=(255,))
        y += line_heights[i]

    array_mask = np.array(img)
    array_img = np.zeros((array_mask.shape[0] // 4 * 4 + 4, array_mask.shape[1] // 4 * 4 + 4, 4), dtype=np.uint8)
    array_img[:array_mask.shape[0], :array_mask.shape[1],:3] = color[:3]
    array_img[:array_mask.shape[0], :array_mask.shape[1],-1] = array_mask
    return array_img