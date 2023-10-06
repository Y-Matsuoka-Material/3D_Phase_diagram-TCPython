from time import ctime
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import matplotlib.pyplot as plt
from .glhelper import gen_text_image
from .video_writer import VideoWriter
from PIL import Image


class GLWindow:
    def __init__(self, xsize, ysize, title, visible=True):
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 9)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        if not visible:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.size = (xsize, ysize)
        self.window = glfw.create_window(*self.size, title, None, None)

        self.rotating = True

        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.lx = 0.0
        self.ly = 0.0
        self.lz = 0.0

        self.tick = -1

        self.back_col = (0., 0., 0.)

        self.text_cache = {}
        self.recorder = None

        glfw.make_context_current(self.window)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_ALPHA_TEST)
        glEnable(GL_DEPTH_TEST)

        glfw.set_key_callback(self.window, self.keyboard)

    def run(self):
        self.tick += 1
        if self.rotating:
            self.ay += 1.0
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        self.key_input()
        if glfw.window_should_close(self.window):
            glfw.terminate()
            return False
        else:
            self.prepare_draw()
            return True

    def prepare_draw(self):
        glClearColor(*list(map(lambda i: i / 255, self.back_col)), 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        limit = -self.lz + 1.5
        glOrtho(-limit, limit, -limit, limit, -64., 64.)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glTranslated(self.lx, self.ly, self.lz)
        glRotated(self.ax, 1.0, 0.0, 0.0)
        glRotated(self.ay, 0.0, 1.0, 0.0)
        glRotated(self.az, 0.0, 0.0, 1.0)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)


    def key_input(self):
        shift = glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT)

        if glfw.get_key(self.window, glfw.KEY_DOWN):
            if shift:
                self.ly -= 0.01
            else:
                self.ax += 2.0
        if glfw.get_key(self.window, glfw.KEY_UP):
            if shift:
                self.ly += 0.01
            else:
                self.ax -= 2.0
        if glfw.get_key(self.window, glfw.KEY_RIGHT):
            if shift:
                self.lx += 0.01
            else:
                self.ay += 2.0
        if glfw.get_key(self.window, glfw.KEY_LEFT):
            if shift:
                self.lx -= 0.01
            else:
                self.ay -= 2.0

        if glfw.get_key(self.window, glfw.KEY_PAGE_UP):
            self.lz += 0.01
        if glfw.get_key(self.window, glfw.KEY_PAGE_DOWN):
            self.lz -= 0.01

    def keyboard(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, 1)
            return
        elif key == glfw.KEY_F2:
            self.take_screenshot()
        elif key == glfw.KEY_SPACE:
            self.rotating = not self.rotating

    def load_text_texture(self, text, color):
        if text in self.text_cache:
            tex_num, size = self.text_cache[text]
        else:
            tex_num = glGenTextures(1)
            im = np.swapaxes(gen_text_image(text, (255 * np.array(color)).astype(np.uint8)), 0, 1)
            glBindTexture(GL_TEXTURE_2D, tex_num)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im.shape[1], im.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, im)
            size = im.shape[:2]
            self.text_cache[text] = (tex_num, size)
        return tex_num, size

    def draw(self, points, mode, color):
        glEnableClientState(GL_VERTEX_ARRAY)
        if len(color) == 3:
            glColor3dv(color)
        elif len(color) == 4:
            glColor4dv(color)
        glVertexPointerf(points)
        glDrawArrays(mode, 0, len(points))

    def draw_with_tex(self, points, mode, texcords, filter=GL_LINEAR, texnum=None):
        if texnum is not None:
            glBindTexture(GL_TEXTURE_2D, texnum)

        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter)

        glColor4dv((1., 1., 1., 1.))

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        glVertexPointerf(points)
        glTexCoordPointerf(texcords)
        glDrawArrays(mode, 0, len(points))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

        glDisable(GL_TEXTURE_2D)

    def draw_text(self, text, center, scl=0.3, rotated=False, alignment="center", color=(1., 1., 1.)):
        tex_num, size = self.load_text_texture(text, color)

        coords = np.array([[0., 0., 0], [1., 0., 0], [1., 1., 0], [0., 1., 0]]) * scl
        coords[:, 0] *= size[0] / size[1]
        texcoords = np.array([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])

        glBindTexture(GL_TEXTURE_2D, tex_num)
        glDepthMask(GL_FALSE)

        glPushMatrix()
        glTranslated(center[0], center[1], center[2])
        glRotated(-self.ay, 0.0, 1.0, 0.0)
        glRotated(-self.ax, 1.0, 0.0, 0.0)
        if rotated: glRotated(90., 0., 0., 1.)
        if alignment == "center":
            glTranslated(-size[0] / size[1] * scl * 1.16 * 0.5, -scl * 0.5, 0.)
        elif alignment == "left":
            glTranslated(0., -scl * 0.5, 0.)
        elif alignment == "right":
            glTranslated(-size[0] / size[1] * 1.16 * scl, -scl * 0.5, 0.)
        self.draw_with_tex(coords, GL_QUADS, texcoords)
        glPopMatrix()

        glDepthMask(GL_TRUE)

    def get_current_image(self, alpha=False):
        if alpha:
            return np.frombuffer(glReadPixels(0, 0, *self.size, GL_RGBA, GL_UNSIGNED_BYTE), dtype=np.uint8).reshape(
                (self.size[1], self.size[0], 4))[::-1]
        else:
            return np.frombuffer(glReadPixels(0, 0, *self.size, GL_RGB, GL_UNSIGNED_BYTE), dtype=np.uint8).reshape(
                (self.size[1], self.size[0], 3))[::-1]

    def take_screenshot(self, name=None, alpha=False):
        name = name or ctime().replace(":", "_").replace(" ", "_")
        name = name.replace(".png", "").replace(".PNG", "")
        Image.fromarray(self.get_current_image(alpha)).save(name + ".png")

    def record(self, filename, fps=60.0):
        if self.recorder is None:
            self.recorder = VideoWriter(filename, fps, self.size)
        self.recorder.write(self.get_current_image()[..., ::-1])

    def set_window_visible(self, flag):
        if flag:
            glfw.show_window(self.window)
        else:
            glfw.hide_window(self.window)
