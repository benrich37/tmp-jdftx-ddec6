import numpy as np
from os.path import join as opj, exists as ope
from ase.io import read
from ase.units import Bohr
from ase import Atoms, Atom
from ase.data import chemical_symbols
from os import environ, chdir, getcwd
from subprocess import run
from scipy.interpolate import RegularGridInterpolator
from time import time
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.ase import AseAtomsAdaptor



def add_redun_layer(d, axis):
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

def sum_d_periodic_grid(d, pbc):
    S_sum = list(np.shape(d))
    for i, v in enumerate(pbc):
        if v:
            S_sum[i] -= 1
    sum_d = np.sum(d[:S_sum[0], :S_sum[1], :S_sum[2]])
    return sum_d

def get_pbc(calc_dir):
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



def write_ddec6_inputs(calc_dir, outname="out", dfname="n", dupfname="n_up", ddnfname="n_dn", data_fname="density", pbc=None, a_d_path=None, max_space=None, norm_density=False, offset=0):
    """
    :param calc_dir: Path to directory containing calc output files needed for ddec6
    :param outname: file name for out file
    :param dfname: file name for total density array
    :param dupfname: file name for spin up density array
    :param ddnfname: file name for spin down density array
    :param data_fname: string to assign XSF and job name (arbitrary)
    :param pbc: List of bools indicating axes which using periodic boundary conition
    :param a_d_path: Path to atomic_densities directory
    :param max_space: Max spacing (in A) between adjacent points on density grid (density grid adjusted via linear interpolation)
    :param norm_density: Normalize to expected electron count
    :return:
    """
    if pbc is None:
        pbc = get_pbc(calc_dir)
    if a_d_path is None:
        a_d_path = a_d_default
    outfile = opj(calc_dir, outname)
    non_col = True # Non_col means spin up and down are separated
    if ope(opj(calc_dir, dfname)):
        non_col = False
    elif not ope(opj(calc_dir, dupfname)):
        raise ValueError("Could not find electron density files")
    if non_col:
        write_ddec6_inputs_noncol(calc_dir, outfile, dupfname, ddnfname, pbc, data_fname, a_d_path, max_space, norm_density=norm_density, offset=offset)
    else:
        write_ddec6_inputs_col(calc_dir, outfile, dfname, pbc, data_fname, a_d_path, max_space)

def write_ddec6_inputs_col(calc_dir, outfile, dfname, pbc, data_fname, a_d_path, max_space):
    atoms = get_atoms(calc_dir)
    _S = get_density_shape(outfile)
    d = get_density_array(calc_dir, _S, dfname)
    if not max_space is None:
        d, S = check_grid(d, atoms, max_space)
    for i in range(3):
        d, S = add_redun_layer(d, i)
    d = get_normed_d(d, atoms, outfile, pbc, S, _S)
    write_xsf(calc_dir, atoms, S, d, data_fname=data_fname)
    write_job_control(calc_dir, atoms, f"{data_fname}.XSF", outfile, pbc, a_d_path)

def write_ddec6_inputs_noncol(calc_dir, outfile, dupfname, ddnfname, pbc, data_fname, a_d_path, max_space, norm_density=False, offset=0):
    atoms = get_atoms(calc_dir)
    _S = get_density_shape(outfile)
    d_up, d_dn = get_density_arrays(calc_dir, _S, dupfname, ddnfname)
    if not max_space is None:
        d_up, S = check_grid(d_up, atoms, max_space)
        d_dn, S = check_grid(d_dn, atoms, max_space)
    for i in range(3):
        d_up, S = add_redun_layer(d_up, i)
        d_dn, S = add_redun_layer(d_dn, i)
    if norm_density:
        d_up, d_dn = get_normed_ds(d_up, d_dn, atoms, outfile, pbc, S, _S, offset_count=offset)
    write_xsf(calc_dir, atoms, S, d_up, d_dn=d_dn, data_fname=data_fname)
    write_job_control(calc_dir, atoms, f"{data_fname}.XSF", outfile, pbc, a_d_path)

def run_ddec6(calc_dir, _exe_path=None):
    if _exe_path is None:
        _exe_path = exe_path
    chdir(calc_dir)
    print(f"Running ddec6 in {calc_dir}")
    run(f"{_exe_path}", shell=True, check=True)

def get_atoms(path):
    if ope(opj(path, "CONTCAR.gjf")):
        atoms = read(opj(path, "CONTCAR.gjf"), format="gaussian-in")
    elif ope(opj(path, "CONTCAR")):
        atoms = read(opj(path, "CONTCAR"), format="vasp")
    else:
        atoms = get_atoms_from_out(opj(path, "out"))
    return atoms

def get_density_shape(outfile):
    start = get_start_line(outfile)
    Sdone = False
    S = None
    for i, line in enumerate(open(outfile)):
        if i > start:
            if (not Sdone) and line.startswith('Chosen fftbox size'):
                S = np.array([int(x) for x in line.split()[-4:-1]])
                Sdone = True
    if not S is None:
        return S
    else:
        raise ValueError(f"Issue finding density array shape 'S' from out file {outfile}")

def get_density_array(calc_dir, S, dfname):
    d = np.fromfile(opj(calc_dir, dfname))
    for i, v in enumerate(d):
        d[i] = max(v, float(0))
    d = d.reshape(S)
    return d

# def get_density_arrays(calc_dir, S, dupfname, ddnfname):
#     d_up = np.fromfile(opj(calc_dir, dupfname))
#     d_dn = np.fromfile(opj(calc_dir, ddnfname))
#     d_up = d_up.reshape(S)
#     d_dn = d_dn.reshape(S)
#     return d_up, d_dn

def get_density_arrays(calc_dir, S, dupfname, ddnfname):
    d_arrs = [
        np.fromfile(opj(calc_dir, dupfname)),
        np.fromfile(opj(calc_dir, ddnfname))
    ]
    for i, d_arr in enumerate(d_arrs):
        for j, v in enumerate(d_arr):
            d_arr[j] = max(v, float(0))
        d_arrs[i] = d_arrs[i].reshape(S)
    return d_arrs

def interp_3d_array(array_in, S_want):
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

def adjust_grid(d, atoms, maxspace, adjust_bools):
    S_cur = np.shape(d)
    S_want = []
    for i in range(3):
        S_i = S_cur[i]
        if adjust_bools[i]:
            S_i = int(np.ceil(np.linalg.norm(atoms.cell[i])/maxspace))
        S_want.append(S_i)
    d = interp_3d_array(d, S_want)
    return d, S_want

def check_grid(d, atoms, maxspace=0.09):
    S = np.shape(d)
    spacings = [np.linalg.norm(atoms.cell[i])/np.shape(d)[i] for i in range(3)]
    adjusts = [s > maxspace for s in spacings]
    if True in adjusts:
        print(f"Density grid (spacings currently {spacings}) too coarse.\n Interpolating density to finer grid with linear interpolation")
        d, S = adjust_grid(d, atoms, maxspace, adjusts)
        print(f"New spacings: {[np.linalg.norm(atoms.cell[i])/np.shape(d)[i] for i in range(3)]}")
    return d, S

def get_normed_d(d, atoms, outfile, pbc, S, _S):
    """
    :param d:  Density array
    :param atoms: ASE atoms object
    :param outfile: path to out file
    :param pbc: List of bools indicating periodic boundary conditions
    :param S: Current shape of d (after adding redundant layers)
    :param _S: Shape of fftbox from JDFTx calculation
    :return:
    """
    tot_zval = get_target_tot_zval(atoms, outfile)
    pix_vol = atoms.get_volume()/(np.prod(np.shape(d))*(Bohr**3))
    sum_d = sum_d_periodic_grid(d, pbc) # excludes final layer for each axis that is periodic (pbc = list of bools)
    # sum_d = np.sum(d)
    d_new = (d*tot_zval/(pix_vol*sum_d*(np.prod(S)/np.prod(_S))))
    return d_new

def get_normed_ds(d_up, d_dn, atoms, outfile, pbc, S, _S, offset_count=0):
    pbc = [True, True, True] # Override in accordance to how density XSF is written
    tot_zval = get_target_tot_zval(atoms, outfile) + offset_count
    # pix_vol = atoms.get_volume() / (np.prod(np.shape(d_up)) * (Bohr ** 3))
    pix_vol = atoms.get_volume() / (np.prod(_S) * (Bohr ** 3))
    sum_d_up = sum_d_periodic_grid(d_up, pbc)
    sum_d_dn = sum_d_periodic_grid(d_dn, pbc)
    sum_d = sum_d_up + sum_d_dn
    coef = (tot_zval / (pix_vol * sum_d * (np.prod(S) / np.prod(_S))))
    d_up_new = d_up*coef
    d_dn_new = d_dn*coef
    return d_up_new, d_dn_new

def write_xsf(calc_dir, atoms, S, d_up, d_dn = None, data_fname="density"):
    xsf_str = make_xsf_str(atoms, S, d_up, d_dn, data_fname)
    xsf_fname = f"{data_fname}.XSF"
    xsf_file = opj(calc_dir, xsf_fname)
    with open(xsf_file, "w") as f:
        f.write(xsf_str)
    f.close()

def write_job_control(calc_dir, atoms, xsf_fname, outfile, pbc, a_d_path):
    print("writing job control")
    nelecs = get_n_elecs(outfile)
    print(f"nelecs: {nelecs}")
    atom_type_count_dict = get_atom_type_count_dict(atoms)
    print(f"atom_type_count_dict: {atom_type_count_dict}")
    atom_types = list(atom_type_count_dict.keys())
    print(f"atom_types: {atom_types}")
    atom_type_core_elecs_dict = get_atom_type_core_elecs_dict(atom_types, outfile)
    print(f"atom_type_core_elecs_dict: {atom_type_core_elecs_dict}")
    elecs_per_atom_type_for_neutral_dict = get_elecs_per_atom_type_for_neutral_dict(atom_type_core_elecs_dict)
    print(f"elecs_per_atom_type_for_neutral_dict: {elecs_per_atom_type_for_neutral_dict}")
    elecs_for_neutral = get_elecs_for_neutral(atom_type_count_dict, elecs_per_atom_type_for_neutral_dict)
    print(f"elecs_for_neutral: {elecs_for_neutral}")
    net_charge = elecs_for_neutral - nelecs
    print(f"net_charge: {net_charge}")
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

def get_target_tot_zval(atoms, outfile):
    nelecs = get_n_elecs(outfile)
    # atom_type_count_dict = get_atom_type_count_dict(atoms)
    # atom_types = list(atom_type_count_dict.keys())
    # Z_vals = [get_Z_val(el, outfile) for el in atom_types]
    # tot_zval = 0
    # for i, el in enumerate(atom_types):
    #     tot_zval += atom_type_count_dict[el]*Z_vals[i]
    return nelecs

def get_n_core_elecs_str(atom_type_core_elecs_dict):
    title = "number of core electrons"
    contents = ""
    for el in list(atom_type_core_elecs_dict.keys()):
        atomic_number = chemical_symbols.index(el)
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
        all_elecs = float(chemical_symbols.index(el))
        req_elecs = all_elecs - atom_type_core_elecs_dict[el]
        elecs_per_atom_type_for_neutral_dict[el] = req_elecs
    return elecs_per_atom_type_for_neutral_dict

def get_atom_type_core_elecs_dict(atom_types, outfile):
    atom_type_core_elecs_dict = {}
    for el in atom_types:
        atom_type_core_elecs_dict[el] = get_atom_type_core_elecs(el, outfile)
    return atom_type_core_elecs_dict

def get_atom_type_core_elecs(el, outfile):
    Z_val = get_valence_electrons(el, outfile)
    core_elecs = float(chemical_symbols.index(el)) - Z_val
    return core_elecs

def get_valence_electrons(el, outfile):
    start_line = get_start_line(outfile)
    reading_key = "Reading pseudopotential file"
    valence_key = " valence electrons"
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
    print(f"Z_val for {el}: {Z_val}")
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

def get_atom_type_count_dict(atoms):
    count_dict = {}
    for el in atoms.get_chemical_symbols():
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

def get_atoms_from_pmg_joutstructure(jstruc):
    jstruc = joutstructures[idx]
    atoms = AseAtomsAdaptor.get_atoms(jstruc.structure)
    E = 0
    if not jstruc.e is None:
        E = jstruc.e
    atoms.E = E
    charges = np.zeros(len(atoms))
    if not jstruc.charges is None:
        for i, charge in enumerate(jstruc.charges):
            charges[i] = charge
    atoms.set_initial_charges(charges)

def get_atoms_list_from_pmg_joutstructures(jstrucs):
    atoms_list = []
    for jstruc in jstrucs:
        atoms = get_atoms_from_pmg_joutstructure(jstruc)
        atoms_list.append(atoms)
    return atoms_list

def get_atoms_list_from_pmg_jdftxoutfileslice(jdftxoutfile_slice):
    jstrucs = jdftxoutfile_slice.joutstructures
    return get_atoms_list_from_pmg_joutstructures(jstrucs)

def get_atoms_list_from_pmg_jdftxoutfile(jdftxoutfile):
    atoms_list = []
    for jdftxoutfile_slice in jdftxoutfile:
        if jdftxoutfile_slice is not None:
            atoms_list += get_atoms_list_from_pmg_jdftxoutfileslice(jdftxoutfile_slice)
        else:
            atoms_list.append(None)
    return atoms_list

def get_atoms_from_out(outfile):
    atoms_list = get_atoms_list_from_out(outfile)
    return atoms_list[-1]


def get_atoms_list_from_out(outfile_path):
    outfile = JDFTXOutfile.from_file(outfile_path, none_slice_on_error=True)
    _atoms_list = get_atoms_list_from_pmg_jdftxoutfile(outfile)
    atoms_list = [a for a in _atoms_list if a is not None]
    return atoms_list



# def get_atoms_list_from_out_slice(outfile, i_start, i_end):
#     charge_key = "oxidation-state"
#     opts = []
#     nAtoms = None
#     R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
#         new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars()
#     for i, line in enumerate(open(outfile)):
#         if i > i_start and i < i_end:
#             if new_posn:
#                 if "Lowdin population analysis " in line:
#                     active_lowdin = True
#                 elif "R =" in line:
#                     active_lattice = True
#                 elif "# Forces in" in line:
#                     active_forces = True
#                     coords_forces = line.split()[3]
#                 elif line.find('# Ionic positions in') >= 0:
#                     coords = line.split()[4]
#                     active_posns = True
#                 elif active_lattice:
#                     if lat_row < 3:
#                         R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
#                         lat_row += 1
#                     else:
#                         active_lattice = False
#                         lat_row = 0
#                 elif active_posns:
#                     tokens = line.split()
#                     if len(tokens) and tokens[0] == 'ion':
#                         names.append(tokens[1])
#                         posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
#                         if tokens[1] not in idxMap:
#                             idxMap[tokens[1]] = []
#                         idxMap[tokens[1]].append(j)
#                         j += 1
#                     else:
#                         posns = np.array(posns)
#                         active_posns = False
#                         nAtoms = len(names)
#                         if len(charges) < nAtoms:
#                             charges = np.zeros(nAtoms)
#                 ##########
#                 elif active_forces:
#                     tokens = line.split()
#                     if len(tokens) and tokens[0] == 'force':
#                         forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
#                     else:
#                         forces = np.array(forces)
#                         active_forces = False
#                 ##########
#                 elif "Minimize: Iter:" in line:
#                     if "F: " in line:
#                         E = float(line[line.index("F: "):].split(' ')[1])
#                     elif "G: " in line:
#                         E = float(line[line.index("G: "):].split(' ')[1])
#                 elif active_lowdin:
#                     if charge_key in line:
#                         look = line.rstrip('\n')[line.index(charge_key):].split(' ')
#                         symbol = str(look[1])
#                         line_charges = [float(val) for val in look[2:]]
#                         chargeDir[symbol] = line_charges
#                         for atom in list(chargeDir.keys()):
#                             for k, idx in enumerate(idxMap[atom]):
#                                 charges[idx] += chargeDir[atom][k]
#                     elif "#" not in line:
#                         active_lowdin = False
#                         log_vars = True
#                 elif log_vars:
#                     if np.sum(R) == 0.0:
#                         R = get_input_coord_vars_from_outfile(outfile)[2]
#                     if coords != 'cartesian':
#                         posns = np.dot(posns, R)
#                     if len(forces) == 0:
#                         forces = np.zeros([nAtoms, 3])
#                     if coords_forces.lower() != 'cartesian':
#                         forces = np.dot(forces, R)
#                     opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
#                     R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
#                         new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars(
#                         nAtoms=nAtoms)
#             elif "Computing DFT-D3 correction:" in line:
#                 new_posn = True
#     return opts

def get_atoms_list_from_out_reset_vars(nAtoms=100, _def=100):
    R = np.zeros([3, 3])
    posns = []
    names = []
    chargeDir = {}
    active_lattice = False
    lat_row = 0
    active_posns = False
    log_vars = False
    coords = None
    new_posn = False
    active_lowdin = False
    idxMap = {}
    j = 0
    E = 0
    if nAtoms is None:
        nAtoms = _def
    charges = np.zeros(nAtoms, dtype=float)
    forces = []
    active_forces = False
    coords_forces = None
    return R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces

def get_atoms_from_outfile_data(names, posns, R, charges=None, E=0, momenta=None):
    atoms = Atoms()
    posns *= Bohr
    R = R.T*Bohr
    atoms.cell = R
    if charges is None:
        charges = np.zeros(len(names))
    if momenta is None:
        momenta = np.zeros([len(names), 3])
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i], momentum=momenta[i]))
    atoms.E = E
    return atoms

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

def make_xsf_str(atoms, S, d_up, d_dn, data_fname):
    dump_str = "CRYSTAL\n"
    dump_str += make_primvec_str(atoms)
    dump_str += make_primcoord_str(atoms)
    dump_str += make_datagrid_str(atoms, d_up, S, data_fname, spin="1")
    if not d_dn is None:
        dump_str += make_datagrid_str(atoms, d_dn, S, data_fname, spin="2")
    return dump_str

def make_datagrid_str(atoms, d, S, data_fname, spin="1"):
    dump_str = "BEGIN_BLOCK_DATAGRID_3D\n"
    dump_str += f" DATA_from:{data_fname}.RHO\n"
    dump_str += f" BEGIN_DATAGRID_3D_RHO:spin_{spin}\n"
    _S = [str(s) for s in S]
    for i in range(3):
        dump_str += " "*(6-len(_S[i])) + _S[i]
    dump_str += "\n"
    dump_str += make_datagrid_str_lattice(atoms)
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
                # ns = f"{d[i,j,k]:.5e}"
                # dump_str += " "*(13-len(ns)) + ns
            dump_str += "\n"
    return dump_str

def make_datagrid_str_lattice(atoms):
    dump_str = ""
    origin = np.zeros(3)
    for j in range(3):
        num_str = f"{origin[j]:.8f}"
        dump_str += " "*(15-len(num_str))
        dump_str += num_str
    dump_str += "\n"
    for i in range(3):
        for j in range(3):
            num_str = f"{atoms.cell[i,j]:.8f}"
            dump_str += " "*(15-len(num_str))
            dump_str += num_str
        dump_str += "\n"
    return dump_str

def make_primvec_str(atoms):
    dump_str = "PRIMVEC\n"
    for i in range(3):
        for j in range(3):
            num_str = f"{atoms.cell[i,j]:.8f}"
            dump_str += " "*(20-len(num_str)) + num_str
        dump_str += "\n"
    return dump_str

def make_primcoord_str(atoms):
    dump_str = "PRIMCOORD\n"
    dump_str += f"   {len(atoms)} 1\n"
    at_nums = atoms.get_atomic_numbers()
    at_nums = [str(n) for n in at_nums]
    # at_nums = [f"{n:.8f}" for n in at_nums]
    posns = atoms.positions
    _posns = []
    for p in posns:
        _posns.append([])
        for i in range(3):
            _posns[-1].append(f"{p[i]:.8f}")
    for i in range(len(atoms)):
        dump_str += " "*(4-len(at_nums[i])) + at_nums[i]
        for j in range(3):
            pstr = _posns[i][j]
            dump_str += " "*(20-len(pstr))
            dump_str += pstr
        dump_str += "\n"
    return dump_str


########## FUNCTION GRAVEYARD ##########
#
# def remove_redun_layer(d, axis):
#     S_new = list(np.shape(d))
#     S_new[axis] -= 1
#     d_new = np.zeros(S_new)
#     for i in range(S_new[0]):
#         for j in range(S_new[1]):
#             for k in range(S_new[2]):
#                 d_new[i, j, k] += d[i, j, k]
#     return d_new, S_new
#
# def print_all_factors(factors, exponents, base=1.0):
#     nf = len(factors)
#     ne = len(exponents)
#     ni = ne**(nf)
#     for i in range(ni):
#         exps = []
#         for f in range(nf):
#             exp_f = int(np.floor((i % (ne**(f+1)))/(ne**f)))
#             exps.append(exp_f)
#         exps = [exponents[exp] for exp in exps]
#         convs = [float(factors[f])**(exps[f]) for f in range(nf)]
#         conv = np.prod(convs)*base
#         print(str(conv) + ": [" + ", ".join([str(exp) for exp in exps]) + "]")
#
########################################

def ran_successfully(calc_dir):
    return ope(opj(calc_dir, "DDEC6_even_tempered_net_atomic_charges.xyz"))


def run_ddec6_runner(calc_dir, a_d_env_path, pbc, exe_env_path, norm=False, offset=0):
    write_ddec6_inputs(calc_dir, max_space=None, a_d_path=a_d_env_path, pbc=pbc, norm_density=norm, offset=offset)
    run_ddec6(calc_dir, _exe_path=exe_env_path)


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


def run_ddec6_looper(calc_dir, a_d_env_path, pbc, exe_env_path):
    success = False
    run_ddec6_runner(calc_dir, a_d_env_path, pbc, exe_env_path)
    success = ran_successfully(calc_dir)
    if success:
        return None
    else:
        offset = 0
        print(f"Run without norm unsuccessful. Attempting with norm offset by {offset}")
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
a_d_default = "/global/cfs/cdirs/m4025/Software/Perlmutter/ddec6/chargemol_09_26_2017/atomic_densities/"
exe_path = "/global/cfs/cdirs/m4025/Software/Perlmutter/ddec6/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_parallel"

# Set these to an environmental variable to override the default strings above
a_d_varname = None
exe_path_varname = None


a_d_key = "DDEC6_AD_PATH"
exe_key = "DDEC6_EXE_PATH"

if not a_d_varname is None:
    a_d_default = environ[a_d_varname]
if not exe_path_varname is None:
    exe_path = environ[exe_path_varname]




def main(calc_dir=None, pbc=None):
    pbc = [True, True, True] # override
    if calc_dir is None:
        calc_dir = getcwd()
    a_d_env_path = None
    exe_env_path = None
    if a_d_key in environ:
        a_d_env_path = environ[a_d_key]
    if exe_key in environ:
        exe_env_path = environ[exe_key]
    # If your fftbox is too coarse, adding max_space=0.1 can force ddec6 to work with a linear interpolation onto
    # a finer density grid.
    run_ddec6_looper(calc_dir, a_d_env_path, pbc, exe_env_path)


if __name__ == "__main__":
    main()
