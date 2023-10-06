# coding: UTF-8
import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import psutil
from tc_python import *
import atexit
import time

# ----------------------------CONFIGURATION PART START----------------------------
# Calculation conditions----------------------------------------------------------

# Elements used in the calculation
elements = ["Ni", "Co", "Cr"]

# List of phases considered in the equilibrium calculation
# If list is empty, the default phases selected by ThermoCalc is used.
considered_phases = []

# Temperature range in Kelvin
T_range = [600, 2400]

# The name of database used in the calculation
# TDB file can be also used by specifying the path to the TDB file.
database = "TCNI11"

# Calculation settings------------------------------------------------------------

# Number of grid divisions
ND = 48

# Number of parallelization
num_core = 12

# The flag determining whether VA and /- are included in the calculation.
# Set to False if the database does not contain VA and /-
add_special_component = True

# Visualization settings----------------------------------------------------------

# Label of the y-axis.
ylabel = f"Temperature (K)"

# Whether to show tielines.
show_tieline = False

# Whether to use different colors for phase separation.
# Set to False if colorization fails.
distinguish_phase_separation = True

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
# -----------------------------CONFIGURATION PART END-----------------------------

is_user_database = database.endswith(".tdb") or database.endswith(".TDB")
database_name = database.replace("/", "\\").split("\\")[-1][:-4] if is_user_database else database

filename_base = f"{'-'.join(elements)}_{','.join(considered_phases)}_{T_range[0]}-{T_range[1]}K_{database_name}"
data_filename = f"data/{filename_base}.npz"
legend_filename = f"img/{filename_base}.png"
video_filename = f"img/{filename_base}.avi"
model_filename = f"img/{filename_base}.glb"

if not os.path.isdir("img"):
    os.mkdir("img")

if not os.path.isdir("data"):
    os.mkdir("data")


# Get conditions of calculation
def get_condition(process_no):
    conds_all = np.zeros((3, ND, 4))
    conds_all[:, :, :3] = (np.arange(3)[np.newaxis, np.newaxis, :] + np.arange(3)[:, np.newaxis, np.newaxis]) % 3
    conds_all[:, :, 3] = np.linspace(1e-6, 1, ND, endpoint=False)
    return conds_all.reshape((ND * 3, 4))[process_no::num_core]


# Preparation of the calculation
def prepare_calculation(session, pd=True):
    if is_user_database:
        if add_special_component:
            elements_with_special = elements + ["/-", "VA"]
        else:
            elements_with_special = elements
        calc = session.select_user_database_and_elements(path_to_user_database=database,
                                                         list_of_elements=elements_with_special)
    else:
        calc = session.select_database_and_elements(database_name=database, list_of_elements=elements)

    if considered_phases != []:
        calc = calc.without_default_phases()
        for m in considered_phases:
            calc = calc.select_phase(m)

    if pd:
        calc_base = calc.get_system().with_phase_diagram_calculation()
    else:
        calc_base = calc.get_system().with_single_equilibrium_calculation()
    return calc_base


# Calculation of phase diagram
def calc_pd(process_no):
    time.sleep(process_no)

    conds = get_condition(process_no)
    result = []
    with TCPython(logging_policy=LoggingPolicy.NONE) as session:
        atexit.register(stop_api_server)

        calc_base = prepare_calculation(session)

        for num, cond in enumerate(conds):
            axis_dependent, axis_x, axis_fixed, c_fixed = cond
            axis_dependent, axis_x, axis_fixed = int(axis_dependent), int(axis_x), int(axis_fixed)
            ind = np.argsort(np.array([axis_dependent, axis_x, axis_fixed, 3]))

            axisname_x, axisname_y, axisname_fixed = f"X({elements[axis_x]})", "T", f"X({elements[axis_fixed]})"

            min_x, max_x = 0, 1
            min_y, max_y = T_range[0], T_range[1]

            calc = (calc_base.remove_all_conditions().
                    with_first_axis(CalculationAxis(axisname_x).set_min(min_x).set_max(max_x)).
                    with_second_axis(CalculationAxis(axisname_y).set_min(min_y).set_max(max_y)).
                    set_condition(f"P", 100000).
                    set_condition(f"N", 1).
                    set_condition(axisname_x, 0.01).
                    set_condition(axisname_y, 1000)
                    )

            try:
                phase_diagram = (calc.set_condition(axisname_fixed, c_fixed).
                                 calculate().
                                 get_values_grouped_by_stable_phases_of(axisname_x, axisname_y)
                                 )

                for key, v in phase_diagram.get_lines().items():
                    cx, T = np.array(v.get_x()), np.array(v.get_y())
                    cz = np.full_like(cx, c_fixed)
                    ca = 1. - cx - cz
                    result.append((key, np.c_[ca, cx, cz, T][:, ind]))

                v = phase_diagram.get_invariants()
                cx, T = np.array(v.get_x()), np.array(v.get_y())
                cz = np.full_like(cx, c_fixed)
                ca = 1. - cx - cz
                result.append(("invariant", np.c_[ca, cx, cz, T][:, ind]))
            except Exception:
                calc_base = prepare_calculation(session)
            finally:
                print(f"Process {process_no}: {num + 1} / {len(conds)}")
    atexit.unregister(stop_api_server)
    print(f"Process {process_no}: Completed.")
    return result

# Calculation of tieline
def calc_tieline(N):
    two_phase = []
    with TCPython(logging_policy=LoggingPolicy.NONE) as start:
        calc_base = (prepare_calculation(start, pd=False).
                     set_condition("P", 100000.0).
                     set_condition("N", 1.0)
                     )
        for num in range(N):
            compositions = np.random.rand(4)
            compositions[:3] /= np.sum(compositions[:3])
            compositions[-1] = T_range[0] + (T_range[1] - T_range[0]) * compositions[-1]
            for e, c in zip(elements[1:], compositions[1:-1]):
                calc_base = calc_base.set_condition(f"X({e})", c)
            T = compositions[-1]
            calc_base = calc_base.set_condition("T", T)

            try:
                calc_result = calc_base.calculate()

                phases = calc_result.get_stable_phases()
                if len(phases) == 2:
                    for p in phases:
                        two_phase.append([calc_result.get_value_of(f"X({p},{e})") for e in elements] + [T])
            except:
                pass
    return np.array(two_phase).reshape((-1, 4))


def calculate():
    with ProcessPoolExecutor(max_workers=num_core) as pool:
        lines = pool.map(calc_pd, list(range(num_core)))
        for child in psutil.Process().children():
            child.nice(psutil.IDLE_PRIORITY_CLASS)
    lines = sum(lines, [])

    phases = {}
    for label, line in lines:
        if label in phases:
            phases[label] = np.concatenate((phases[label], np.full((1, 4), np.nan), line), axis=0)
        else:
            phases[label] = line

    with ProcessPoolExecutor(max_workers=num_core) as pool:
        tielines = pool.map(calc_tieline,
                            [1000 // num_core] * num_core)
        for child in psutil.Process().children():
            child.nice(psutil.IDLE_PRIORITY_CLASS)
    tielines = list(tielines)
    tieline_two = np.concatenate(tielines, axis=0)
    np.savez(data_filename, **{k: v for k, v in phases.items()}, tieline_two=tieline_two)
    data = np.load(data_filename)
    phases = {k: data[k] for k in sorted(data.files)}
    return data, phases


def prepare_verts():
    cmap_ = plt.get_cmap(cmap)

    if os.path.isfile(data_filename):
        data = np.load(data_filename)
        phases = {k: data[k] for k in sorted(data.files)}
    else:
        data, phases = calculate()

    tieline = phases.pop("tieline_two")

    if not distinguish_phase_separation:
        phases_new = {}
        for k_, v in phases.items():
            k = k_.split("#")[0]
            if k in phases_new:
                phases_new[k] = np.concatenate((phases_new[k], np.full((1, 4), np.nan), v), axis=0)
            else:
                phases_new[k] = v
        phases = phases_new

    # ------------------------------------------------------------------------------------------------------------------

    p = np.concatenate((triangle, [[0., 1., 0.]]), axis=0)

    ii = 0
    verts_pd = {}
    for k, v in phases.items():
        if k == "invariant":
            col = (1., 1., 1., 1.)
        else:
            col = cmap_(ii % 10)
            ii += 1

        v[:, 3] = (v[:, 3] - T_range[0]) / (T_range[1] - T_range[0]) * 2 - 1
        verts_pd[k] = (v @ p, col)

    tieline[:, 3] = (tieline[:, 3] - T_range[0]) / (T_range[1] - T_range[0]) * 2 - 1
    verts_tieline = tieline @ p
    return verts_pd, verts_tieline


# Display and save legend window
def show_legend_window():
    for k, v in verts_pd.items():
        if v[0].size == 0 or np.all(np.isnan(v[0])):
            continue
        plt.plot(np.nan, np.nan, label=k, color=v[1])

    plt.legend(ncol=2, prop={'family': 'Arial', "size": 15})
    plt.axis("off")
    plt.savefig(legend_filename, transparent=True)
    plt.pause(1 / 30)

# Saving results to the 3D model file
def save_GLTF():
    gltf = gltfutil.GLTFModel()
    gltf.add_lines(verts_frame)

    for k, v in verts_pd.items():
        if np.all(np.isnan(v[0])):
            continue
        gltf.add_lines_strip(v[0], color=v[1])

    if show_tieline:
        gltf.add_lines(verts_tieline, color=(0., 1.0, 0.))

    gltf.save(model_filename.replace(".glb", "_notext.glb"))

    for y_v, y_p in zip(ytick_values, yticks_pos):
        gltf.add_text(f"{y_v}", y_p, (y_p[0], 0., y_p[2]))
    gltf.add_text(ylabel, triangle[0] * 1.1 - 0.25 * (triangle[1] - triangle[2]),
                  facing=(yticks_pos[0, 0], 0., yticks_pos[0, 2]), up=(-yticks_pos[0, 2], 0., yticks_pos[0, 0]))
    labelcenter = np.c_[triangle[:, 0], np.full(3, -1.2), triangle[:, 2]]
    labelfacing = labelcenter.copy()
    labelfacing[:, 1] = 0.
    for i in range(3):
        gltf.add_text(elements[i], labelcenter[i], (-yticks_pos[0, 0], 0., -yticks_pos[0, 2]))

    gltf.save(model_filename)

# Show interactive window
def show_interactive_window():
    gl = glutil.GLWindow(512, 512, "Ternary phase diagram", visible=not save_animation)
    gl.back_col = background_color
    gl.ax, gl.ay, gl.az, gl.lx, gl.ly, gl.lz = 20.0, 0, 0.0, 0.0, 0.0, -0.15

    while gl.run():
        gl.draw(verts_frame, glutil.GL_LINES, color=np.array(frame_color) / 255.0)

        for k, v in verts_pd.items():
            gl.draw(v[0], glutil.GL_LINE_STRIP, v[1])

        if show_tieline:
            gl.draw(verts_tieline, glutil.GL_LINES, (0., 1.0, 0.))

        for name, pos in zip(elements, triangle):
            gl.draw_text(name, pos - [0, 1.2, 0], color=np.array(text_color) / 255.0)

        for y_v, y_p in zip(ytick_values, yticks_pos):
            gl.draw_text(f"{y_v}", y_p, color=np.array(text_color) / 255)

        gl.draw_text(ylabel, triangle[0] * 1.35, rotated=True, color=np.array(text_color) / 255)

        if save_animation and gl.tick < 360:
            gl.record(video_filename)
            if gl.tick == 359:
                gl.set_window_visible(True)


if __name__ == '__main__':
    from utils import glutil
    from utils import gltfutil
    from utils import ticks
    from utils import trimesh

    triangle = np.c_[trimesh.triangle[:, 0], np.full(3, 0), trimesh.triangle[:, 1]]

    verts_pd, verts_tieline = prepare_verts()

    lims, ytick_values = ticks.get_nice_lim_and_ticks(T_range[0], T_range[1], 0.5, T_range[0], T_range[1])
    yticks_pos = np.repeat([triangle[0]], ytick_values.shape[0], 0) * 1.1
    yticks_pos[:, 1] = (ytick_values - lims[0]) / (lims[1] - lims[0]) * 2 - 1

    verts_frame = trimesh.gen_frame()

    if save_3dmodel:
        save_GLTF()

    show_legend_window()
    show_interactive_window()
