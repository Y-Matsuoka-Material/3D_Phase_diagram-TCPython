# coding: UTF-8
import os
import numpy as np
import matplotlib.pyplot as plt
from tc_python import *
from utils import glutil
from utils import gltfutil
from utils import ticks
from utils import trimesh

# ----------------------------CONFIGURATION PART START----------------------------
# Calculation conditions----------------------------------------------------------

# Elements used in the calculation
elements = ["Ni", "Co", "Cr"]

# Specification of phase.
# Phase equilibria can be considered by specifying multiple phases in inner list.
phases_list = [["BCC_B2"], ["FCC_L12"]]

# Temperature in Kelvin
T = 1273

# The name of database used in the calculation
# TDB file can be also used by specifying the path to the TDB file.
database = "TCNI11"

# Calculation settings------------------------------------------------------------

# Number of grid divisions
ND = 51

# The flag determining whether VA and /- are included in the calculation.
# Set to False if the database does not contain VA and /-
add_special_component = True

# Variable name of the target and label of the y-axis.
target_variable, ylabel, scaling = "G", "Gibbs energy (kJ/mol)", 1e-3
# target_variable, ylabel, scaling = "H", "Enthalpy (kJ/mol)", 1e-3

# Visualization settings----------------------------------------------------------

# lower and upper limit of y-axis
draw_min, draw_max = None, None  # If set to None, limits is automatically set.
# draw_min, draw_max = -500, 0

# Colormap for the mesh.
# Colormaps from matplotlib can be used.
cmap = "tab10"

# Whether to save animation or not.
save_animation = True

# Whether to save 3D model or not.
save_3dmodel = True

background_color = (0, 0, 0)  # Color of the background.
frame_color = (255, 255, 255)  # Color of the plot frame.
text_color = (255, 255, 255)  # Color of the text.

# --------------------------------------------------------------------------------

is_user_database = database.endswith(".tdb") or database.endswith(".TDB")
database_name = database.replace("/", "\\").split("\\")[-1][:-4] if is_user_database else database

filename_base = f"{'-'.join(elements)}_{T}K_{database_name}_{target_variable}"
legend_filename = f"img/{filename_base}.png"
video_filename = f"img/{filename_base}.avi"
model_filename = f"img/{filename_base}.glb"

if not os.path.isdir("img"):
    os.mkdir("img")

# Preparation of the calculation
def prepare_calculation(session):
    if is_user_database:
        if add_special_component:
            elements_with_special = elements + ["/-", "VA"]
        else:
            elements_with_special = elements
        calc_base = session.select_user_database_and_elements(path_to_user_database=database,
                                                              list_of_elements=elements_with_special)
    else:
        calc_base = session.select_database_and_elements(database_name=database, list_of_elements=elements)

    for phases in phases_list:
        for phase in phases:
            calc_base = calc_base.select_phase(phase)

    calc_base = calc_base.get_system().with_batch_equilibrium_calculation()
    return calc_base

# Calculation of Gibbs energy
def calculate():
    def func(c1, c2, c3):
        nonlocal calc
        calc = (calc.
                set_condition("P", 100000.0).
                set_condition("N", 1.0).
                set_condition("T", T).
                disable_global_minimization()
                )

        flag = (0 <= c1) & (c1 <= 1) & (0 <= c2) & (c2 <= 1) & (0 <= c3) & (c3 <= 1)
        G = np.full_like(c2, np.nan)

        calc.set_conditions_for_equilibria(
            [[(f"X({elements[1]})", xx), (f"X({elements[2]})", yy), ] for xx, yy in
             zip(c2[flag], c3[flag])])

        calc_result = calc.calculate([target_variable], logging_frequency=0)

        G[flag] = np.array(calc_result.get_values_of(target_variable))
        return G * scaling

    verts_list = []
    with TCPython() as session:
        calc = prepare_calculation(session)

        for phases in phases_list:
            if phases == []:
                calc = calc.set_phase_to_entered(ALL_PHASES)
            else:
                calc = calc.set_phase_to_suspended(ALL_PHASES)
                for phase in phases:
                    calc = calc.set_phase_to_entered(phase)

            verts = trimesh.gen_verts_raw(func, ND)
            verts_list.append(verts)
    return verts_list

# Display and save legend window
def show_legend_window():
    for phases, col in zip(phases_list, cols_list):
        if phases == []:
            label = "ALL PHASES"
        else:
            label = " & ".join(phases)
        plt.plot(np.nan, np.nan, label=label, color=col)

    plt.legend(ncol=2, prop={'family': 'Arial', "size": 15})
    plt.axis("off")
    plt.savefig(legend_filename, transparent=True)
    plt.pause(1 / 30)

# Saving results to the 3D model file
def save_GLTF():
    gltf = gltfutil.GLTFModel()
    gltf.add_lines(verts_frame)

    for verts, col in zip(verts_list, cols_list):
        gltf.add_lines_strip(verts, color=col)

    gltf.save(model_filename.replace(".glb", "_notext.glb"))

    for y_v, y_p in zip(ytick_values, yticks_pos):
        gltf.add_text(f"{y_v}", y_p, (y_p[0], 0., y_p[2]))
    gltf.add_text(ylabel, triangle[0] * 1.1 - 0.15 * (triangle[1] - triangle[2]),
                  facing=(yticks_pos[0, 0], 0., yticks_pos[0, 2]), up=(-yticks_pos[0, 2], 0., yticks_pos[0, 0]))
    labelcenter = np.c_[trimesh.triangle[:, 0], np.full(3, -1), trimesh.triangle[:, 1]]
    labelfacing = labelcenter.copy()
    labelfacing[:, 1] = 0.
    for i in range(3):
        gltf.add_text(elements[i], labelcenter[i], (-yticks_pos[0, 0], 0., -yticks_pos[0, 2]))

    gltf.save(model_filename)

# Show interactive window
def show_interactive_window():
    gl = glutil.GLWindow(512, 512, "Energy surface", visible=not save_animation)
    gl.back_col = background_color
    gl.ax, gl.ay, gl.az, gl.lx, gl.ly, gl.lz = 20.0, 0, 0.0, 0.0, 0.0, -0.25

    while gl.run():
        gl.draw(verts_frame, glutil.GL_LINES, (1., 1., 1.))

        for verts, col in zip(verts_list, cols_list):
            gl.draw(verts, glutil.GL_LINE_STRIP, col)

        for t, name in zip(triangle, elements):
            gl.draw_text(name, t + [0, -1.2, 0], color=np.array(text_color) / 255)

        for y_v, y_p in zip(ytick_values, yticks_pos):
            gl.draw_text(f"{y_v}", y_p, color=np.array(text_color) / 255)

        gl.draw_text(ylabel, triangle[0] * 1.35, rotated=True, color=np.array(text_color) / 255)

        if save_animation and gl.tick < 360:
            gl.record(video_filename)
            if gl.tick == 359:
                gl.set_window_visible(True)


if __name__ == '__main__':
    triangle = np.c_[trimesh.triangle[:, 0], np.full(3, 0), trimesh.triangle[:, 1]]

    # Calculation of gibbs energy
    verts_list = calculate()

    lims, ytick_values = ticks.get_nice_lim_and_ticks(min(map(lambda v: np.nanmin(v[:, 1]), verts_list)),
                                                      max(map(lambda v: np.nanmax(v[:, 1]), verts_list)),
                                                      0.4, draw_min, draw_max)
    yticks_pos = np.repeat([triangle[0]], ytick_values.shape[0], 0) * 1.1
    yticks_pos[:, 1] = (ytick_values - lims[0]) / (lims[1] - lims[0]) * 2 - 1

    cols_list = []
    for i, verts in enumerate(verts_list):
        verts[:, 1] = (verts[:, 1] - lims[0]) / (lims[1] - lims[0]) * 2 - 1
        cols_list.append(plt.get_cmap(cmap)(i))

    verts_frame = trimesh.gen_frame()

    # Save 3D model file
    if save_3dmodel:
        save_GLTF()

    # Interactive visualization
    show_legend_window()
    show_interactive_window()
