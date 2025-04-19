import numpy as np
from os.path import join as opj, exists as ope
from os import environ, chdir, getcwd
from subprocess import run
from scipy.interpolate import RegularGridInterpolator
from time import time
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.core.units import bohr_to_ang

# TODO:
# - replace ASE with pymatgen
# - replace xsf with CHGCAR
# - interface with pymatgen.commands.ddec



def add_redun_layer(d: np.ndarray, axis: int):
    S_old = list(np.shape(d))
    S_new = list(np.shape(d))
    S_new[axis] += 1
    d_new = np.zeros(S_new)
    for i in range(S_old[0]):
        for j in range(S_old[1]):
            for k in range(S_old[2]):
                d_new[i,j,k] += d[i,j,k]
    axis = axis % 3
    if axis == 0:
        d_new[-1,:,:] += d[0,:,:]
    elif axis == 1:
        d_new[:,-1,:] += d[:,0,:]
    elif axis == 2:
        d_new[:,:,-1] += d[:,:,0]
    return d_new, S_new

def sum_d_periodic_grid(d: np.ndarray, pbc: list[bool]):
    S_sum = list(np.shape(d))
    for i, v in enumerate(pbc):
        if v:
            S_sum[i] -= 1
    sum_d = np.sum(d[:S_sum[0], :S_sum[1], :S_sum[2]])
    return sum_d

def get_pbc(calc_dir: str):
    infname = opj(calc_dir, "in")
    lookkey = "coulomb-interaction"
    tokens = None
    with open(infname, "r") as f:
        for line in f:
            if lookkey in line:
                tokens = line.strip().split()
    if not tokens is None:
        ctype = tokens[1].lower()
        if ctype == "periodic":
            return [True, True, True]
        elif ctype == "isolated":
            return [False, False, False]
        else:
            direction = tokens[2]
            pbc = [v == "0" for v in direction]
            if ctype == "slab":
                return pbc
            elif ctype == "wire":
                pbc = [not v for v in pbc]
                return pbc
            else:
                raise ValueError(f"Coulomb type {ctype} as found from {tokens} not yet supported for pbc detection.")


def find_file_name(calc_dir: str, suffix: str, prefix: str):
    filename = f"{prefix}{suffix}"
    file_path = opj(calc_dir, f"{filename}")
    if not ope(file_path):
        filename = suffix
        file_path = opj(calc_dir, f"{filename}")
    if not ope(file_path):
        return None
    return filename
        



def write_ddec6_inputs(
        calc_dir: str, 
        data_fname: str = "density", 
        pbc: list[bool] | None = None, 
        a_d_env_path: str = None, 
        max_space=None, 
        norm_density=False, 
        offset=0,
        file_prefix=""
        ):
    """
        Write required input files for running chargemol for ddec6 analysis..

        Args:
            calc_dir (str): The path to the directory containing the JDFTx out files.
            data_fname (str): Name of xsf file to write density as. Arbitrary, but changable incase "density.xsf"
                breaks something for you.
            pbc (list[bool, bool, bool]): List of the periodic boundary conditions for the calculation. 
            a_d_env_path (str): The path to the atomic density files.
            max_space: I don't remember
            offset (float | int): Offset to add to density array(s). Made non-zero if default normalization isn't
                good enough for little ol chargemol.
            file_prefix (str): Prefix for jdftx output files `out` and `n`/`n_up`/`n_dn` (ie put "jdftx." if your out file is
                "jdftx.out"). Leave as empty string if no prefix.
    """
    outname = find_file_name(calc_dir, "out", file_prefix)
    if outname is None:
        raise ValueError("Could not find out file")
    dfname = find_file_name(calc_dir, "n", file_prefix)
    dupfname = find_file_name(calc_dir, "n_up", file_prefix)
    ddnfname = find_file_name(calc_dir, "n_dn", file_prefix)
    print(dfname)
    print(dupfname)
    if pbc is None:
        pbc = get_pbc(calc_dir)
    if a_d_env_path is None:
        a_d_env_path = a_d_default
    outfile = opj(calc_dir, outname)
    has_spin = True
    if dfname is None:
        has_spin = True
    elif dupfname is None:
        raise ValueError("Could not find electron density files")
    if has_spin:
        write_ddec6_inputs_spin(calc_dir, outfile, dupfname, ddnfname, pbc, data_fname, a_d_env_path, max_space, norm_density=norm_density, offset=offset)
    else:
        write_ddec6_inputs_nospin(calc_dir, outfile, dfname, pbc, data_fname, a_d_env_path, max_space, norm_density=norm_density, offset=offset)

def write_ddec6_inputs_nospin(
        calc_dir: str, outfile_path: str, dfname: str, pbc: list[bool], 
        data_fname: str, a_d_path: str, max_space: float | None,
        norm_density=False, offset: float = 0.0
        ):
    jof = JDFTXOutfile.from_file(outfile_path)
    structure = jof.structure
    _S = get_density_shape(outfile_path)
    d = get_density_arrays(calc_dir, _S, [dfname])[0]
    if not max_space is None:
        d, S = check_grid(d, structure, max_space)
    for i in range(3):
        d, S = add_redun_layer(d, i)
    if norm_density:
        d = get_normed_d(d, structure, outfile_path, pbc, S, _S, offset_count=offset)
    write_xsf(calc_dir, structure, S, d, data_fname=data_fname)
    write_job_control(calc_dir, structure, f"{data_fname}.XSF", outfile_path, pbc, a_d_path)

def write_ddec6_inputs_spin(
        calc_dir: str, outfile_path: str, dupfname: str, ddnfname: str, 
        pbc: list[bool], data_fname: str, a_d_path: str, max_space: list[bool], 
        norm_density=False, offset: float = 0.0
        ):
    jof = JDFTXOutfile.from_file(outfile_path)
    structure = jof.structure
    _S = get_density_shape(outfile_path)
    d_up, d_dn = get_density_arrays(calc_dir, _S, [dupfname, ddnfname])
    if not max_space is None:
        d_up, S = check_grid(d_up, structure, max_space)
        d_dn, S = check_grid(d_dn, structure, max_space)
    for i in range(3):
        d_up, S = add_redun_layer(d_up, i)
        d_dn, S = add_redun_layer(d_dn, i)
    if norm_density:
        d_up, d_dn = get_normed_ds(d_up, d_dn, structure, outfile_path, pbc, S, _S, offset_count=offset)
    write_xsf(calc_dir, structure, S, d_up, d_dn=d_dn, data_fname=data_fname)
    write_job_control(calc_dir, structure, f"{data_fname}.XSF", outfile_path, pbc, a_d_path)

def run_ddec6(calc_dir: str, exe_path: str):
    chdir(calc_dir)
    print(f"Running ddec6 in {calc_dir}")
    run(f"{exe_path}", shell=True, check=True)


def get_density_shape(outfile_path: str):
    start = get_start_line(outfile_path)
    Sdone = False
    S = None
    for i, line in enumerate(open(outfile_path)):
        if i > start:
            if (not Sdone) and line.startswith('Chosen fftbox size'):
                S = np.array([int(x) for x in line.split()[-4:-1]])
                Sdone = True
    if not S is None:
        return S
    else:
        raise ValueError(f"Issue finding density array shape 'S' from out file {outfile_path}")


def get_density_arrays(calc_dir: str, S: list[int], dfnames: list[str]):
    d_arrs = [np.fromfile(opj(calc_dir, dfname)) for dfname in dfnames]
    for i, d_arr in enumerate(d_arrs):
        d_arrs[i] = correct_density_for_negative(d_arr)
        d_arrs[i] = d_arrs[i].reshape(S)
    return d_arrs

def correct_density_for_negative(d: np.ndarray):
    prevsum = np.sum(d.flatten())
    d = np.maximum(d, np.zeros(len(d)))
    d *= prevsum/np.sum(d.flatten())
    return d

def interp_3d_array(array_in, S_want):
    """ Return an interpolated density array.

    Return an interpolated density array.

    Args:
        array_in (np.ndarray): Density array of shape S
        S_want (list[int]): Densired shape of density array.
    """
    S_cur = np.shape(array_in)
    cx = np.linspace(0, 1, S_cur[0])
    cy = np.linspace(0, 1, S_cur[1])
    cz = np.linspace(0, 1, S_cur[2])
    wx = np.linspace(0, 1, S_want[0])
    wy = np.linspace(0, 1, S_want[1])
    wz = np.linspace(0, 1, S_want[2])
    # pts = np.meshgrid(wx, wy, wz, indexing='ij', sparse=True)
    start = time()
    pts_shape = S_want
    pts_shape.append(3)
    pts = np.zeros(pts_shape)
    for i in range(S_want[0]):
        pts[i, :, :, 0] += wx[i]
    for i in range(S_want[1]):
        pts[:, i, :, 1] += wy[i]
    for i in range(S_want[1]):
        pts[:, :, i, 2] += wz[i]
    end = time()
    print(f"getting pts: {end - start}")
    interp = RegularGridInterpolator((cx, cy, cz), array_in)
    start = time()
    new_array = interp(pts)
    end = time()
    print(f"interpolating: {end - start}")
    return new_array

def adjust_grid(d, structure: Structure, maxspace, adjust_bools):
    S_cur = np.shape(d)
    S_want = []
    cell = structure.lattice.matrix
    for i in range(3):
        S_i = S_cur[i]
        if adjust_bools[i]:
            S_i = int(np.ceil(np.linalg.norm(cell[i])/maxspace))
        S_want.append(S_i)
    d = interp_3d_array(d, S_want)
    return d, S_want

def check_grid(d, structure: Structure, maxspace=0.09):
    """ Check density grid for adequate spacing.

    Check density grid for adequate spacing. Returns an interpolated grid with finer spacing if
    grid does not pass criteria set in maxspace.

    Args:
        d (np.ndarray): Electron density array of shape S.
        structure (Structure): Pymatgen Structure object of structure used in jdftx calculation.
            Needed to get lattice vector lengths.
        maxspace (float): Maximum length in Anstroms allowed for a voxel.
    """
    cell = structure.lattice.matrix
    S = np.shape(d)
    spacings = [np.linalg.norm(cell[i])/np.shape(d)[i] for i in range(3)]
    adjusts = [s > maxspace for s in spacings]
    if True in adjusts:
        print(f"Density grid (spacings currently {spacings}) too coarse.\n Interpolating density to finer grid with linear interpolation")
        d, S = adjust_grid(d, structure, maxspace, adjusts)
        print(f"New spacings: {[np.linalg.norm(cell[i])/np.shape(d)[i] for i in range(3)]}")
    return d, S

# def get_normed_d(d, structure: Structure, outfile_path: str, pbc: list[bool], S: list[int], _S: list[int]):
#     tot_zval = get_target_tot_zval(structure, outfile_path)
#     pix_vol = structure.lattice.volume/(np.prod(np.shape(d))*(bohr_to_ang**3))
#     sum_d = sum_d_periodic_grid(d, pbc) # excludes final layer for each axis that is periodic (pbc = list of bools)
#     # sum_d = np.sum(d)
#     d_new = (d*tot_zval/(pix_vol*sum_d*(np.prod(S)/np.prod(_S))))
#     return d_new

def get_normed_ds(d_up, d_dn, structure: Structure, outfile, pbc, S, _S, offset_count=0):
    tot_zval = get_target_tot_zval(structure, outfile) + offset_count
    pix_vol = structure.lattice.volume / (np.prod(_S) * (bohr_to_ang ** 3))
    sum_d_up = sum_d_periodic_grid(d_up, pbc)
    sum_d_dn = sum_d_periodic_grid(d_dn, pbc)
    sum_d = sum_d_up + sum_d_dn
    print(f"tot_zval: {tot_zval}")
    print(f"pix_vol: {pix_vol}")
    print(f"sum_d: {sum_d}")
    print(f"prod_S: {np.prod(S)}")
    print(f"prod_S_: {np.prod(_S)}")
    coef = (tot_zval / (pix_vol * sum_d * (np.prod(S) / np.prod(_S))))
    print(f"coef: {coef}")
    d_up_new = d_up*coef
    d_dn_new = d_dn*coef
    return d_up_new, d_dn_new

def get_normed_d(d, structure: Structure, outfile_path: str, pbc: list[bool], S: list[int], _S: list[int], offset_count=0):
    tot_zval = get_target_tot_zval(structure, outfile_path) + offset_count
    pix_vol = structure.lattice.volume / (np.prod(_S) * (bohr_to_ang ** 3))
    sum_d = sum_d_periodic_grid(d, pbc)
    print(f"tot_zval: {tot_zval}")
    print(f"pix_vol: {pix_vol}")
    print(f"sum_d: {sum_d}")
    print(f"prod_S: {np.prod(S)}")
    print(f"prod_S_: {np.prod(_S)}")
    coef = (tot_zval / (pix_vol * sum_d * (np.prod(S) / np.prod(_S))))
    print(f"coef: {coef}")
    d = d*coef
    return d

def write_xsf(calc_dir, structure: Structure, S, d_up, d_dn = None, data_fname="density"):
    xsf_str = make_xsf_str(structure, S, d_up, d_dn, data_fname)
    xsf_fname = f"{data_fname}.XSF"
    xsf_file = opj(calc_dir, xsf_fname)
    with open(xsf_file, "w") as f:
        f.write(xsf_str)
    f.close()

def write_job_control(calc_dir, structure: Structure, xsf_fname, outfile, pbc, a_d_path):
    nelecs = get_n_elecs(outfile)
    atom_type_count_dict = get_atom_type_count_dict(structure)
    atom_types = list(atom_type_count_dict.keys())
    atom_type_core_elecs_dict = get_atom_type_core_elecs_dict(atom_types, outfile)
    elecs_per_atom_type_for_neutral_dict = get_elecs_per_atom_type_for_neutral_dict(atom_type_core_elecs_dict)
    elecs_for_neutral = get_elecs_for_neutral(atom_type_count_dict, elecs_per_atom_type_for_neutral_dict)
    net_charge = elecs_for_neutral - nelecs
    job_control_str = get_job_control_str(net_charge, pbc, xsf_fname, atom_type_core_elecs_dict, a_d_path)
    with open(opj(calc_dir, "job_control.txt"), "w") as f:
        f.write(job_control_str)
    f.close()

def write_job_control_alt(calc_dir, structure: Structure, xsf_fname, outfile_path, pbc, a_d_path):
    # Alternate version that only uses JDFTXOutfile data
    outfile = JDFTXOutfile.from_file(outfile_path)
    #nelecs = get_n_elecs(outfile_path)
    nelecs = outfile.total_electrons
    atom_type_count_dict = get_atom_type_count_dict(structure)
    atom_types = list(atom_type_count_dict.keys())
    # TODO: Write in the "valence_electrons_uncharged_dict" into the jdftxoutfileslice
    atom_type_core_elecs_dict = get_atom_type_core_elecs_dict(atom_types, outfile_path)
    #atom_type_core_elecs_dict = get_atom_type_core_elecs_dict_alt(outfile)
    #elecs_per_atom_type_for_neutral_dict = get_elecs_per_atom_type_for_neutral_dict(atom_type_core_elecs_dict)
    #elecs_for_neutral = get_elecs_for_neutral(atom_type_count_dict, elecs_per_atom_type_for_neutral_dict)
    elecs_for_neutral = outfile.total_electrons_uncharged
    net_charge = elecs_for_neutral - nelecs
    job_control_str = get_job_control_str(net_charge, pbc, xsf_fname, atom_type_core_elecs_dict, a_d_path)
    with open(opj(calc_dir, "job_control.txt"), "w") as f:
        f.write(job_control_str)
    f.close()

#####################

def get_job_control_str(net_charge, pbc, xsf_fname, atom_type_core_elecs_dict, a_d_path):
    dump_str = ""
    dump_str += get_net_charge_str(net_charge)
    dump_str += get_periodicity_str(pbc)
    dump_str += get_atomic_densities_str(a_d_path)
    dump_str += get_input_fname_str(xsf_fname)
    dump_str += get_charge_type_str("DDEC6")
    dump_str += get_n_core_elecs_str(atom_type_core_elecs_dict)
    return dump_str

def get_target_tot_zval(structure, outfile):
    nelecs = get_n_elecs(outfile)
    return nelecs

def get_n_core_elecs_str(atom_type_core_elecs_dict):
    title = "number of core electrons"
    contents = ""
    for el in list(atom_type_core_elecs_dict.keys()):
        atomic_number = Element(el).Z
        contents += f"{atomic_number} {int(atom_type_core_elecs_dict[el])}\n"
    return get_job_control_piece_str(title, contents)

def get_charge_type_str(charge_type):
    title = "charge type"
    contents = charge_type
    return get_job_control_piece_str(title, contents)

def get_input_fname_str(xsf_fname):
    title = "input filename"
    contents = str(xsf_fname)
    return get_job_control_piece_str(title, contents)

def get_atomic_densities_str(a_d_path):
    title = "atomic densities directory complete path"
    contents = a_d_path
    return get_job_control_piece_str(title, contents)

def get_periodicity_str(pbc):
    title = "periodicity along A, B, and C vectors"
    contents = ""
    for v in pbc:
        vstr = "false"
        if v:
            vstr = "true"
        contents += f".{vstr}.\n"
    return get_job_control_piece_str(title, contents)

def get_net_charge_str(net_charge):
    title = "net charge"
    contents = str(net_charge)
    return get_job_control_piece_str(title, contents)

def get_job_control_piece_str(title, contents):
    contents = contents.rstrip("\n")
    dump_str = f"<{title}>\n{contents}\n</{title}>\n\n"
    return dump_str

############################################

def get_elecs_for_neutral(atom_type_count_dict, elecs_per_atom_type_for_neutral_dict):
    elecs_for_neutral = 0
    for el in list(atom_type_count_dict.keys()):
        elecs_for_neutral += elecs_per_atom_type_for_neutral_dict[el]*atom_type_count_dict[el]
    return elecs_for_neutral

def get_elecs_per_atom_type_for_neutral_dict(atom_type_core_elecs_dict):
    elecs_per_atom_type_for_neutral_dict = {}
    for el in list(atom_type_core_elecs_dict.keys()):
        all_elecs = float(Element(el).Z)
        req_elecs = all_elecs - atom_type_core_elecs_dict[el]
        elecs_per_atom_type_for_neutral_dict[el] = req_elecs
    return elecs_per_atom_type_for_neutral_dict

def get_atom_type_core_elecs_dict(atom_types, outfile: str):
    atom_type_core_elecs_dict = {}
    for el in atom_types:
        atom_type_core_elecs_dict[el] = get_atom_type_core_elecs(el, outfile)
    return atom_type_core_elecs_dict

def get_atom_type_core_elecs_dict_alt(outfile: JDFTXOutfile):
    atom_type_core_elecs_dict = {}
    for el in atom_types:
        atom_type_core_elecs_dict[el] = get_atom_type_core_elecs(el, outfile)
    return atom_type_core_elecs_dict

def get_atom_type_core_elecs(el, outfile):
    Z_val = get_Z_val(el, outfile)
    core_elecs = float(Element(el).Z) - Z_val
    return core_elecs

def get_Z_val(el, outfile):
    start_line = get_start_line(outfile)
    reading_key = "Reading pseudopotential file"
    valence_key = " valence electrons in orbitals"
    Z_val = 0
    with open(outfile, "r") as f:
        reading = False
        for i, line in enumerate(f):
            if i > start_line:
                if reading:
                    if valence_key in line:
                        v = line.split(valence_key)[0].split(" ")[-1]
                        Z_val = int(v)
                        break
                    else:
                        continue
                else:
                    if reading_key in line:
                        fpath = line.split("'")[1]
                        fname = fpath.split("/")[-1]
                        ftitle = fname.split(".")[0].lower()
                        if el.lower() in ftitle.split("_"):
                            reading = True
                        else:
                            reading = False
    return Z_val

def get_n_elecs(outfile):
    nelecs_key = "nElectrons: "
    with open(outfile, "r") as f:
        for line in f:
            if nelecs_key in line:
                nelec_line = line
    v = nelec_line.split(nelecs_key)[1].strip().split(" ")[0]
    nelecs = float(v)
    return nelecs

def get_atom_type_count_dict(structure):
    count_dict = {}
    for site in structure:
        el = site.specie.symbol
        if not el in count_dict:
            count_dict[el] = 0
        count_dict[el] += 1
    return count_dict

def get_start_lines(outfname, add_end=False):
    start_lines = []
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line or "Input parsed successfully" in line:
            start_lines.append(i)
        end_line = i
    if add_end:
        start_lines.append(end_line)
    return start_lines

def get_start_line(outfile):
    start_lines = get_start_lines(outfile, add_end=False)
    return start_lines[-1]


def get_input_coord_vars_from_outfile(outfname):
    start_line = get_start_line(outfname)
    names = []
    posns = []
    R = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    with open(outfname) as f:
        for i, line in enumerate(f):
            if i > start_line:
                tokens = line.split()
                if len(tokens) > 0:
                    if tokens[0] == "ion":
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    elif tokens[0] == "lattice":
                        active_lattice = True
                    elif active_lattice:
                        if lat_row < 3:
                            R[lat_row, :] = [float(x) for x in tokens[:3]]
                            lat_row += 1
                        else:
                            active_lattice = False
                    elif "Initializing the Grid" in line:
                        break
    return names, posns, R

def make_xsf_str(structure, S, d_up, d_dn, data_fname):
    dump_str = "CRYSTAL\n"
    dump_str += make_primvec_str(structure)
    dump_str += make_primcoord_str(structure)
    dump_str += make_datagrid_str(structure, d_up, S, data_fname, spin="1")
    if not d_dn is None:
        dump_str += make_datagrid_str(structure, d_dn, S, data_fname, spin="2")
    return dump_str

def make_datagrid_str(structure, d, S, data_fname, spin="1"):
    dump_str = "BEGIN_BLOCK_DATAGRID_3D\n"
    dump_str += f" DATA_from:{data_fname}.RHO\n"
    dump_str += f" BEGIN_DATAGRID_3D_RHO:spin_{spin}\n"
    _S = [str(s) for s in S]
    for i in range(3):
        dump_str += " "*(6-len(_S[i])) + _S[i]
    dump_str += "\n"
    dump_str += make_datagrid_str_lattice(structure)
    dump_str += make_datagrid_str_dens(d, S)
    dump_str += " END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n"
    return dump_str

def make_datagrid_str_dens(d, S):
    dump_str = ""
    for k in range(S[2]):
        for j in range(S[1]):
            for i in range(S[0]):
                ns = f"{d[i, j, k]:.8e}"
                dump_str += " " + ns
            dump_str += "\n"
    return dump_str

def make_datagrid_str_lattice(structure: Structure):
    dump_str = ""
    origin = np.zeros(3)
    for j in range(3):
        num_str = f"{origin[j]:.8f}"
        dump_str += " "*(15-len(num_str))
        dump_str += num_str
    dump_str += "\n"
    for i in range(3):
        for j in range(3):
            num_str = f"{structure.lattice.matrix[i,j]:.8f}"
            dump_str += " "*(15-len(num_str))
            dump_str += num_str
        dump_str += "\n"
    return dump_str

def make_primvec_str(structure):
    dump_str = "PRIMVEC\n"
    for i in range(3):
        for j in range(3):
            num_str = f"{structure.lattice.matrix[i,j]:.8f}"
            dump_str += " "*(20-len(num_str)) + num_str
        dump_str += "\n"
    return dump_str

def make_primcoord_str(structure: Structure):
    dump_str = "PRIMCOORD\n"
    dump_str += f"   {len(structure)} 1\n"
    at_nums = structure.atomic_numbers
    at_nums = [str(n) for n in at_nums]
    posns = structure.cart_coords
    _posns: list[list] = []
    for p in posns:
        _posns.append([])
        for i in range(3):
            _posns[-1].append(f"{p[i]:.8f}")
    for i in range(len(structure)):
        dump_str += " "*(4-len(at_nums[i])) + at_nums[i]
        for j in range(3):
            pstr = _posns[i][j]
            dump_str += " "*(20-len(pstr))
            dump_str += pstr
        dump_str += "\n"
    return dump_str


def ran_successfully(calc_dir):
    return ope(opj(calc_dir, "DDEC6_even_tempered_net_atomic_charges.xyz"))


def run_ddec6_runner(calc_dir: str, a_d_env_path: str, pbc: list[bool], exe_env_path: str, norm: bool = False, offset: float | int = 0, file_prefix: str = ""):
    """
        Write required input files and run chargemol for ddec6 analysis..

        Args:
            calc_dir (str): The path to the directory containing the JDFTx out files.
            a_d_env_path (str): The path to the atomic density files.
            pbc (list[bool, bool, bool]): List of the periodic boundary conditions for the calculation. 
            exe_env_path (str): The path to the chargemol executable.
            file_prefix (str): Prefix for jdftx output files `out` and `n`/`n_up`/`n_dn`. Leave as empty string if
                no prefix.
    """
    write_ddec6_inputs(calc_dir, max_space=None, a_d_env_path=a_d_env_path, pbc=pbc, norm_density=norm, offset=offset, file_prefix=file_prefix)
    run_ddec6(calc_dir, exe_env_path)


def get_ddec6_output_nvalence(density_output):
    key = "nvalence ="
    with open(density_output, "r") as f:
        for line in f:
            if key in line:
                val = float(line.split("=")[1].strip())
                return val

def get_ddec6_output_integ_valence(density_output):
    key = "numerically integrated valence density ="
    with open(density_output, "r") as f:
        for line in f:
            if key in line:
                val = float(line.split("=")[1].strip())
                return val


def get_checkme(calc_dir):
    density_output = opj(calc_dir, "density.output")
    nvalance = get_ddec6_output_nvalence(density_output)
    integrated = get_ddec6_output_integ_valence(density_output)
    print(f"nval: {nvalance}")
    print(f"integ: {integrated}")
    checkme = integrated - nvalance
    return checkme


def adjust_offset(offset, calc_dir):
    checkme = get_checkme(calc_dir)
    offset -= checkme
    return offset



def run_ddec6_looper(calc_dir: str, a_d_env_path: str, pbc: list[bool], exe_env_path: str, file_prefix: str = ""):
    """
        Loop the function `run_ddec6_runner`, allowing for adjustable normalization offsets incase of incorrect density sum.

        Args:
            calc_dir (str): The path to the directory containing the JDFTx out files.
            a_d_env_path (str): The path to the atomic density files. (Note - this MUST end with "/" (or whatever delimiter
                is used by the file system), otherwise chargemol will look for files named "atomic_densitiesc2_033..."
                instead of "atomic_densities/c2_033...").
            pbc (list[bool, bool, bool]): List of the periodic boundary conditions for the calculation. Used to inform
                where to add redundant layering.
            exe_env_path (str): The path to the chargemol executable.
            file_prefix (str): Prefix for jdftx output files `out` and `n`/`n_up`/`n_dn` (ie put "jdftx." if your out file is
                "jdftx.out"). Leave as empty string if no prefix.
    """
    success = False
    run_ddec6_runner(calc_dir, a_d_env_path, pbc, exe_env_path, file_prefix=file_prefix)
    success = ran_successfully(calc_dir)
    if success:
        return None
    else:
        offset = 0
        print(f"Run without norm unsuccessful. Attempting with norm offset starting at {offset}")
        for i in range(3):
            run_ddec6_runner(calc_dir, a_d_env_path, pbc, exe_env_path, norm=True, offset=offset)
            success = ran_successfully(calc_dir)
            if success:
                return None
            else:
                print(f"Run unsuccessful with offset={offset}. Re-evaluating offset")
                offset = adjust_offset(offset, calc_dir)
                print(f"Rerunning with offset={offset}")

pbc_default = [True, True, True]

a_d_key = "DDEC6_AD_PATH"
exe_key = "DDEC6_EXE_PATH"


def main(calc_dir: str = None, a_d_env_path: str = None, exe_env_path: str = None, file_prefix: str = ""):
    pbc = [True, True, True] # Hard-coding the periodic boundary condition to all True for now
    # Hypothetically, the pbc should inform the program which axes to add redundant layering to.
    # However, I've only achieved successful runs without guess-normalizing by setting all to True.
    # This needs to be tested and refined more, because it still feels like I did something wrong with that,
    # but the difference in results is pretty minimal so I'm leaving it as so for now.
    if calc_dir is None:
        calc_dir = getcwd()
    if a_d_key in environ:
        a_d_env_path = environ[a_d_key]
    if exe_key in environ:
        exe_env_path = environ[exe_key]
    # If your fftbox is too coarse, adding max_space=0.1 can force ddec6 to work with a linear interpolation onto
    # a finer density grid.
    run_ddec6_looper(calc_dir, a_d_env_path, pbc, exe_env_path, file_prefix=file_prefix)

if __name__ == "__main__":
    main()
