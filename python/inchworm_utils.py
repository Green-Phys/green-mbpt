import numpy as np
from green_mbtools.pesto.ir import IR_factory
from scipy import interpolate
import scipy.linalg as LA


def read_hopping_from_txt(hopping_file: str, nao: int, uhf: bool = True):
    """_summary_

    Parameters
    ----------
    hopping_file : str
        Path to hopping file
    nao : int
        Number of atomic orbitals
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    np.ndarray
        Hopping matrix in UHF format if uhf is True, else in RHF/GHF format
    """
    if uhf:
        number_of_orbitals = 2 * nao
    hopping_raw = np.zeros((number_of_orbitals, number_of_orbitals), dtype=complex)
    with open(hopping_file) as k:
        for raw_line in k:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            i_str, m_str, ij_str_re, ij_str_im = line.split()
            i = int(i_str)
            m = int(m_str)
            hopping_raw[i, m] = complex(float(ij_str_re) + 1j * float(ij_str_im))
    if uhf:
        hopping_uhf = np.zeros((2, nao, nao), dtype=complex)
        hopping_uhf[0, :, :] = hopping_raw[0::2, 0::2]
        hopping_uhf[1, :, :] = hopping_raw[1::2, 1::2]
    else:
        hopping_uhf = hopping_raw.reshape((1, ) + hopping_raw.shape)
    return hopping_uhf


def read_delta_tau_from_txt(delta_file: str, nao: int, beta: float, uhf: bool = True):
    """Read hybridization function from text file.

    Parameters
    ----------
    delta_file : str
        Path to delta file
    nao : int
        Number of atomic orbitals
    beta : float
        Inverse temperature
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # ---------- pass 1: determine max l = even tau grid index ----------
    l_max = -1
    with open(delta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue  # or raise, if you prefer strict
            l = int(parts[0])
            if l > l_max:
                l_max = l

    if l_max < 0:
        raise ValueError(f"No data lines found in {delta_file}")
    n_tau = l_max + 1

    # ---------- allocate ----------
    if uhf:
        ns = 2
    else:
        ns = 1
    delta_tau = np.zeros((n_tau, ns, nao, nao), dtype=complex)

    # ---------- pass 2: fill ----------
    with open(delta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            l_str, i_str, m_str, re_str, im_str = line.split()
            l = int(l_str)
            i = int(i_str)
            m = int(m_str)
            s1_idx = i % ns
            ao_i = i // ns
            s2_idx = m % ns
            ao_m = m // ns
            if s1_idx != s2_idx:
                continue  # skip spin-flip terms in UHF mode
            delta_tau[l, s1_idx, ao_i, ao_m] = float(re_str) + 1j * float(im_str)

    tau_delta = np.linspace(0.0, beta, n_tau, endpoint=True)

    return delta_tau, tau_delta


def read_greenfunction_from_txt(nao: int, time_filename: str, green_path: str, uhf=True):
    """Read green's function from text files.

    Parameters
    ----------
    nao : int
        Number of atomic orbitals
    time_filename : str
        Path to file containing time points
    green_path : str
        Directory containing green's function files
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the green's function array and the time array
    """
    if uhf:
        number_of_orbitals = 2 * nao
        ns = 2
    else:
        number_of_orbitals = nao
        ns = 1

    # ---------- read time points ----------
    t_list = []
    with open(time_filename) as k:
        for line in k:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # t_str, ij_str = line.split()
            t_str = line.split()[0]
            t_str = t_str.strip().strip('[],')   # remove [ ] and commas
            if t_str:                            # skip empty tokens
                t_list.append(float(t_str))
    t_arr = np.array(t_list)    
    t_shape = len(t_arr)
    
    # ---------- allocate green's function array ----------
    green_tau = np.zeros((t_shape, ns, nao, nao), dtype=complex)
    for i in range(number_of_orbitals):
        for j in range(number_of_orbitals):
            # data = np.loadtxt(f'/home/orit/VS_codes1/example/G_{i}_{j}.dat')
            i_ao = i // ns
            j_ao = j // ns
            s_i = i % ns
            s_j = j % ns
            if s_i != s_j:
                continue # skip spin-flip terms in UHF mode
            with open(f'{green_path}/G_{i}_{j}.dat') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    t_str, ij_str = line.split()            # split into 2 parts
                    # TODO: Waiting for confirmation
                    green_tau[:, s_i, i_ao, j_ao] = np.array([complex(x) for x in ij_str.strip().split(',')])
    return green_tau, t_arr


def interpolation(tau_grid_in, X_tau, tau_grid_new, kind="cubic"):
    """Interpolate X(tau) from tau_grid_in to tau_grid_new
    Parameters
    ----------
    tau_grid_in : np.ndarray
        Original tau grid
    X_tau : np.ndarray
        Data to be interpolated, with tau as the first dimension
    tau_grid_new : np.ndarray
        New tau grid
    kind : str, optional
        Interpolation kind, by default "cubic"
    
    Returns
    -------
    np.ndarray
        Interpolated data on new tau grid
    """
    tau_grid_in = np.asarray(tau_grid_in, dtype=float)
    ntau_in = len(tau_grid_in)
    tau_grid_new = np.asarray(tau_grid_new, dtype=float)
    ntau_out = len(tau_grid_new)
    X_tau = np.asarray(X_tau)

    # merge indices except tau and interpolate
    orig_shape = X_tau.shape
    X_tau = X_tau.reshape((ntau_in, -1))

    # initialize output array
    new_delta_tau = np.zeros((ntau_out, X_tau.shape[-1]), dtype=complex)

    for m in range(X_tau.shape[1]):
        real_interp = interpolate.interp1d(
            tau_grid_in, X_tau[:, m].real,
            kind=kind, fill_value="extrapolate", assume_sorted=True
        )
        imag_interp = interpolate.interp1d(
            tau_grid_in, X_tau[:, m].imag,
            kind=kind, fill_value="extrapolate", assume_sorted=True
        )
        new_delta_tau[:, m] = real_interp(tau_grid_new) + 1j * imag_interp(tau_grid_new)
    
    # reshape to original shape with new tau dimension
    new_delta_tau = new_delta_tau.reshape((ntau_out,) + orig_shape[1:])

    return new_delta_tau


def get_inchworm_selfenergy(green_path, time_file, hopping_file, delta_file, nao_imp, ir_file, beta, mu, uhf=True):
    """Extract static and dynamic self-energies on IR grid from, e.g., Inchworm data

    Parameters
    ----------
    green_path : str
        Path to directory containing green's function files
    time_file : str
        Path to file containing time points
    hopping_file : str
        Path to hopping file
    delta_file : str
        Path to delta file
    nao_imp : int
        Number of impurity orbitals
    ir_file : str
        Path to IR file
    beta : float
        Inverse temperature
    mu : float
        Chemical potential
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    _type_
        _description_
    """

    # Set up IR grid
    myir = IR_factory(beta, ir_file)
    tau_grid_ir = myir.tau_mesh

    green_tau, tau_grid_green = read_greenfunction_from_txt(nao_imp, time_file, green_path, uhf=uhf)
    delta_tau, tau_grid_delta = read_delta_tau_from_txt(delta_file, nao_imp, beta, uhf=uhf)
    hopping = read_hopping_from_txt(hopping_file, nao_imp, uhf=uhf)

    # Interpolate to IR grid
    delta_tau_IR = interpolation(tau_grid_delta, delta_tau, tau_grid_ir, kind="cubic")
    green_tau_IR = interpolation(tau_grid_green, green_tau, tau_grid_ir, kind="cubic")

    # Fourier transform to Matsubara frequency
    delta_omega_IR = myir.tau_to_w(delta_tau_IR)
    green_omega_IR = myir.tau_to_w(green_tau_IR)

    # Compute self-energy in Matsubara frequency
    ns = 2 if uhf else 1
    sigma_omega_IR = np.zeros_like(green_omega_IR, dtype=complex)
    for s in range(ns):
        for iw, w_val in enumerate(myir.wsample):
            g0_inv = (1j * w_val + mu) * np.eye(nao_imp, dtype=complex) - hopping[s] - delta_omega_IR[iw, s]
            g_inv = LA.inv(green_omega_IR[iw, s])
            sigma_omega_IR[iw, s] = g0_inv - g_inv

    # Extract Sigma_inf = sigma (iw_max)
    sigma_inf = sigma_omega_IR[-1]
    for iw, w_val in enumerate(myir.wsample):
        sigma_omega_IR[iw] -= sigma_inf

    # Fourier transform
    sigma_tau_IR = myir.w_to_tau(sigma_omega_IR)

    return sigma_inf, sigma_tau_IR
