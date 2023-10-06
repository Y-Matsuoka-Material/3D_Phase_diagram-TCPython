# coding: UTF-8
import numpy as np
from matplotlib.cm import get_cmap
import pygltflib
from pygltflib.validator import summary, validate
from . import glhelper
import warnings

class GLTFModel:
    def __init__(self):
        self.gltf = pygltflib.GLTF2(scene=0)
        self.scene_start = 0
        self.ind_start = 0
        self.bin_start = 0
        self.material_start = 0
        self.image_start = 0
        self.texture_start = 0
        self.sampler_start = 0
        self.blob = b""

    def add_triangles(self, points, norms=None, inds=None, color=(1., 1., 1., 1.), transparent=False,
                      double_sided=False, emissive=False, uv=None, texture=None, filter="NEAREST"):
        points, norms, inds, uv = glhelper.sanitize_primitive(points, norms, inds, uv)

        if points.size == 0:
            return

        use_norm = norms is not None and not transparent
        use_texture = uv is not None and texture is not None

        maxin = np.nanmax(points, axis=0)
        minin = np.nanmin(points, axis=0)

        if np.any(np.isnan(maxin)) or np.any(np.isnan(minin)):
            return

        points = points.astype(np.float32)
        if use_norm:
            norms = norms.astype(np.float32)
        if use_texture:
            uv = uv.astype(np.float32)

        self.scene_start += 1

        indtype = pygltflib.UNSIGNED_INT

        if points.shape[0] < 2 ** 8:
            inds = inds.astype(np.uint8)
            indtype = pygltflib.UNSIGNED_BYTE
        elif points.shape[0] < 2 ** 16:
            inds = inds.astype(np.uint16)
            indtype = pygltflib.UNSIGNED_SHORT
        else:
            inds = inds.astype(np.uint32)

        points_binary_blob = points.tobytes()
        if use_norm:
            norms_binary_blob = norms.tobytes()
        inds_binary_blob = inds.tobytes()
        if use_texture:
            uv_binary_blob = uv.tobytes()

        self.gltf.nodes.append(pygltflib.Node(mesh=len(self.gltf.meshes)))

        ind_ind = self.ind_start
        self.ind_start += 1
        ind_pos = self.ind_start
        self.ind_start += 1
        if use_norm:
            ind_norm = self.ind_start
            self.ind_start += 1
        else:
            ind_norm = None
        if use_texture:
            ind_tex = self.ind_start
            self.ind_start += 1
        else:
            ind_tex = None

        self.gltf.meshes.append(pygltflib.Mesh(
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=ind_pos,
                                                    NORMAL=ind_norm,
                                                    TEXCOORD_0=ind_tex),
                    mode=pygltflib.TRIANGLES,
                    indices=ind_ind,
                    material=self.material_start
                )
            ]
        ))

        self.gltf.accessors.append(pygltflib.Accessor(
            bufferView=len(self.gltf.bufferViews),
            componentType=indtype,
            count=inds.size,
            type=pygltflib.SCALAR,
            max=[int(inds.max())],
            min=[int(inds.min())],
        ))
        self.gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=self.bin_start,
            byteLength=len(inds_binary_blob),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        ))
        self.bin_start += len(inds_binary_blob)

        self.gltf.accessors.append(pygltflib.Accessor(
            bufferView=len(self.gltf.bufferViews),
            componentType=pygltflib.FLOAT,
            count=len(points),
            type=pygltflib.VEC3,
            max=maxin.tolist(),
            min=minin.tolist(),
        ))
        self.gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=self.bin_start,
            byteLength=len(points_binary_blob),
            target=pygltflib.ARRAY_BUFFER,
        ))
        self.bin_start += len(points_binary_blob)

        if use_norm:
            self.gltf.accessors.append(pygltflib.Accessor(
                bufferView=len(self.gltf.bufferViews),
                componentType=pygltflib.FLOAT,
                count=len(norms),
                type=pygltflib.VEC3,
                max=None,
                min=None,
            ))
            self.gltf.bufferViews.append(pygltflib.BufferView(
                buffer=0,
                byteOffset=self.bin_start,
                byteLength=len(norms_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ))
            self.bin_start += len(norms_binary_blob)

        if use_texture:
            self.gltf.accessors.append(pygltflib.Accessor(
                bufferView=len(self.gltf.bufferViews),
                componentType=pygltflib.FLOAT,
                count=len(uv),
                type=pygltflib.VEC2,
                max=np.nanmax(uv, axis=0).tolist(),
                min=np.nanmin(uv, axis=0).tolist(),
            ))
            self.gltf.bufferViews.append(pygltflib.BufferView(
                buffer=0,
                byteOffset=self.bin_start,
                byteLength=len(uv_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ))
            self.bin_start += len(uv_binary_blob)

        bytelength = len(inds_binary_blob) + len(points_binary_blob)
        if use_norm:
            bytelength += len(norms_binary_blob)
        if use_texture:
            bytelength += len(uv_binary_blob)

        self.gltf.buffers.append(pygltflib.Buffer(
            byteLength=bytelength
        ))
        self.blob += inds_binary_blob
        self.blob += points_binary_blob
        if use_norm:
            self.blob += norms_binary_blob

        if use_texture:
            self.blob += uv_binary_blob

        self.material_start += 1
        material = glhelper.gen_material(color, transparent, double_sided, emissive,
                                            texIndex=self.texture_start if use_texture else None)
        self.gltf.materials.append(material)
        if use_texture:
            filt = pygltflib.NEAREST if filter == "NEAREST" else None
            self.gltf.samplers.append(pygltflib.Sampler(interpolation=filt, magFilter=filt,
                                                        minFilter=filt, wrapS=pygltflib.MIRRORED_REPEAT,
                                                        wrapT=pygltflib.MIRRORED_REPEAT))
            # self.gltf.samplers.append(pygltflib.Sampler())
            self.gltf.textures.append(pygltflib.Texture(sampler=self.sampler_start, source=self.texture_start))
            self.sampler_start += 1
            self.texture_start += 1
            self.gltf.images.append(glhelper.gen_image(texture))

    def flush(self):
        self.gltf.set_binary_blob(self.blob)
        self.gltf.scenes = [pygltflib.Scene(nodes=list(range(self.scene_start))[::-1])]

    def save(self, filename):
        self.flush()
        name = filename.replace(".glb", "").replace(".gltf", "").replace(".GLB", "").replace(".GLTF", "") + ".glb"
        with open(name, "wb") as f:
            f.write(b"".join(self.gltf.save_to_bytes()))

    def add_lines(self, points, inds=None, color=(0., 0., 0., 1.), linewidth=0.01, uv=None, texture=None,
                  filter="LINEAR"):
        if inds is None:
            inds = np.arange(points.shape[0] // 2 * 2)
        i2 = 2 * inds
        im = i2[::2][:, np.newaxis]
        ip = i2[1::2][:, np.newaxis]
        ii = np.concatenate((im, im + 1, ip, im + 1, ip + 1, ip), axis=1).flatten()
        for vec in np.eye(3):
            v = vec * 0.5 * linewidth
            p0 = points + v[np.newaxis, :]
            p1 = points - v[np.newaxis, :]
            pp = np.concatenate((p0, p1), axis=1).reshape((-1, 3))
            if uv is None or texture is None:
                self.add_triangles(pp, None, ii, color, False, True)
            else:
                uvuv = np.concatenate((uv, uv), axis=1).reshape((-1, 2))
                self.add_triangles(pp, None, ii, (1., 1., 1., 1.), False, True, uv=uvuv, texture=texture, filter=filter)

    def add_lines_strip(self, points, color=(0., 0., 0., 1.)):
        ii = np.arange(points.shape[0] - 1)[:, np.newaxis]
        inds = np.concatenate((ii, ii + 1), axis=1).flatten()
        self.add_lines(points, inds, color)

    def add_text(self, text, center=(0., 1.1, 0.), facing=(1., 0., 0.), up=(0., 1., 0.), size=1., color=(0, 0, 0),
                 double_sided=False):
        img = glhelper.gen_text_image(text, color=color)[::-1, ::-1]
        img = np.swapaxes(img, 0, 1)
        x = np.cross(facing, up)
        x = x / np.sum(x ** 2) ** 0.5
        y = np.cross(x, facing)
        y = y / np.sum(y ** 2) ** 0.5
        y *= img.shape[1] / img.shape[0]
        verts = np.array([-x - y,
                          -x + y,
                          +x + y,
                          +x - y
                          ])

        verts *= img.shape[0] / 1500.0
        verts *= size
        verts[:] += center
        uv = np.array([[0., 0.],
                       [1., 0.],
                       [1., 1.],
                       [0., 1.]
                       ])
        inds = np.array([0, 1, 2, 0, 2, 3])
        self.add_triangles(verts, inds=inds, double_sided=double_sided, uv=uv, texture=img, filter="LINEAR")
