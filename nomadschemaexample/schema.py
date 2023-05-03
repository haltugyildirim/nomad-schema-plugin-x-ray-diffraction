from nomad.metainfo import Quantity, Package, SubSection, MEnum
from nomad.datamodel.data import EntryData, ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.eln import Measurement
from nomad.units import ureg
import numpy as np
from xrd_parser import parse_and_convert_file


m_package = Package()


def calculate_two_theta_or_scattering_vector(q=None, two_theta=None, wavelength=None):
    """
    Calculate the two-theta array from the scattering vector (q) or vice-versa,
    given the wavelength of the X-ray source.

    Args:
        q (array-like, optional): Array of scattering vectors, in angstroms^-1.
        two_theta (array-like, optional): Array of two-theta angles, in degrees.
        wavelength (float): Wavelength of the X-ray source, in angstroms.

    Returns:
        numpy.ndarray: Array of two-theta angles, in degrees.
    """
    if q is not None:
        return 2 * np.arcsin(q * wavelength / (4 * np.pi))
    elif two_theta is not None:
        return (4 * np.pi / wavelength) * np.sin(np.deg2rad(two_theta) / 2)
    else:
        raise ValueError("Either q or two_theta must be provided.")


class XRayConventionalSource(ArchiveSection):
    '''
    X-ray source used in conventional diffractometers
    '''
    def estimate_kalpha_wavelengths(self, source_material):
        """
        Estimate the K-alpha1 and K-alpha2 wavelengths of an X-ray source given the material of the source.

        Args:
            source_material (str): Material of the X-ray source, such as 'Cu', 'Fe', 'Mo', 'Ag', 'In', 'Ga', etc.

        Returns:
            Tuple[float, float]: Estimated K-alpha1 and K-alpha2 wavelengths of the X-ray source, in angstroms.
        """
        # Dictionary of K-alpha1 and K-alpha2 wavelengths for various X-ray source materials, in angstroms
        kalpha_wavelengths = {
            'Cr': (2.2910, 2.2936),
            'Fe': (1.9359, 1.9397),
            'Cu': (1.5406, 1.5444),
            'Mo': (0.7093, 0.7136),
            'Ag': (0.5594, 0.5638),
            'In': (0.6535, 0.6577),
            'Ga': (1.2378, 1.2443)
        }

        try:
            kalpha1_wavelength, kalpha2_wavelength = kalpha_wavelengths[source_material]
        except KeyError:
            raise ValueError("Unknown X-ray source material.")

        return kalpha1_wavelength, kalpha2_wavelength

    xray_tube_material = Quantity(
        type=MEnum(sorted(['Cu', 'Cr', 'Mo', 'Fe', 'Ag', 'In', 'Ga'])),
        description='Type of the X-ray tube',
        default='Cu',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.EnumEditQuantity,
        ))

    xray_tube_current = Quantity(
        type=np.dtype(np.float64),
        unit='A',
        description='Current of the X-ray tube',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            label='Current of the X-ray tube'))

    xray_tube_voltage = Quantity(
        type=np.dtype(np.float64),
        unit='V',
        description='Voltage of the X-ray tube',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
            label='Voltage of the X-ray tube'))

    kalpha_one = Quantity(
        type=np.dtype(np.float64),
        unit='angstrom',
        description='Wavelength of the Kα1 line')

    kalpha_two = Quantity(
        type=np.dtype(np.float64),
        unit='angstrom',
        description='Wavelength of the Kα2 line')

    ratio_kalphatwo_kalphaone = Quantity(
        type=np.dtype(np.float64),
        description='Kα2/Kα1 intensity ratio')

    kbeta = Quantity(
        type=np.dtype(np.float64),
        unit='angstrom',
        description='Wavelength of the Kβ line')

    def normalize(self, archive, logger):
        super(XRayConventionalSource, self).normalize(archive, logger)

        if self.xray_tube_material is not None:
            self.kalpha_one, self.kalpha_two = self.estimate_kalpha_wavelengths(self.xray_tube_material)


class XRayDiffraction(Measurement):
    '''
    X-ray diffraction is a technique typically used to characterize the structural
    properties of crystalline materials. The data contains `two_theta` values of the scan
    the corresponding counts collected for each channel
    '''

    def derive_n_values(self):
        if self.intensity is not None:
            return len(self.intensity)
        if self.two_theta is not None:
            return len(self.two_theta)
        else:
            return 0

    method = Quantity(
        type=str,
        description='Method used to collect the data',
        default='X-Ray Diffraction (XRD)')

    n_values = Quantity(type=int, derived=derive_n_values)

    two_theta = Quantity(
        type=np.dtype(np.float64), shape=['n_values'],
        unit='deg',
        description='The 2-theta range of the difractogram',
        a_plot={
            'x': 'two_theta', 'y': 'intensity'
        })

    q_vector = Quantity(
        type=np.dtype(np.float64), shape=['n_values'],
        unit='meter**(-1)',
        description='The scattering vector *Q* of the difractogram',
        a_plot={
            'x': 'q_vector', 'y': 'intensity'
        })

    intensity = Quantity(
        type=np.dtype(np.float64), shape=['n_values'],
        description='The count at each 2-theta value, dimensionless',
        a_plot={
            'x': 'two_theta', 'y': 'intensity'
        })

    omega = Quantity(
        type=np.dtype(np.float64), shape=['n_values'],
        unit='deg',
        description='The omega range of the difractogram')

    phi = Quantity(
        type=np.dtype(np.float64), shape=['n_values'],
        unit='deg',
        description='The phi range of the difractogram')

    chi = Quantity(
        type=np.dtype(np.float64), shape=['n_values'],
        unit='deg',
        description='The chi range of the difractogram')

    source_peak_wavelength = Quantity(
        type=np.dtype(np.float64),
        unit='angstrom',
        description='''Wavelength of the X-ray source. Used to convert from 2-theta to Q
        and vice-versa.''')

    scan_axis = Quantity(
        type=str,
        description='Axis scanned')

    integration_time = Quantity(
        type=np.dtype(np.float64),
        unit='s',
        shape=['*'],
        description='Integration time per channel')

    def normalize(self, archive, logger):
        super(XRayDiffraction, self).normalize(archive, logger)
        try:
            if self.source_peak_wavelength is not None and self.q_vector is not None:
                self.two_theta = calculate_two_theta_or_scattering_vector(
                    q=self.q_vector, wavelength=self.source_peak_wavelength)

            elif self.source_peak_wavelength is not None and self.two_theta is not None:
                self.q_vector = calculate_two_theta_or_scattering_vector(
                    two_theta=self.two_theta, wavelength=self.source_peak_wavelength)
        except Exception:
            logger.warning("Unable to convert from two_theta to q_vector vice-versa")


class XRayDiffractionWithSource(XRayDiffraction):

    source = SubSection(section_def=XRayConventionalSource)

    def normalize(self, archive, logger):
        # this normalize section should copy the kalpha_one into the source_peak_wavelength

        try:
            if self.source.kalpha_one is not None:
                self.source_peak_wavelength = self.source.kalpha_one
            else:
                logger.warning("Unable to set source_peak_wavelegth because source.kalpha_one is None")
        except Exception:
            logger.warning("Unable to set source_peak_wavelegth")

        super(XRayDiffractionWithSource, self).normalize(archive, logger)


class GenericXRD(XRayDiffractionWithSource, EntryData):
    '''
    Generic X-ray diffraction measurement.
    '''

    data_file = Quantity(
        type=str,
        description='Data file containing the difractogram',
        a_eln=dict(
            component='FileEditQuantity',
        )
    )

    def normalize(self, archive, logger):
        super(GenericXRD, self).normalize(archive, logger)

        # Use the xrd parser to populate the schema reading the data file
        if not self.data_file:
            return

        with archive.m_context.raw_file(self.data_file) as file:
            xrd_dict = parse_and_convert_file(file.name)
            self.intensity = xrd_dict['counts']
            self.two_theta = xrd_dict['2Theta'] * ureg('degree')
            self.omega = xrd_dict['Omega'] * ureg('degree')
            # self.kalpha_one = xrd_dict['kAlpha1']
            # self.kalpha_two = xrd_dict['kAlpha2']
            # self.ratio_kalphatwo_kalphaone = xrd_dict['kAlphaRatio']
            # self.kbeta = xrd_dict['kBeta']
            # self.scan_axis = xrd_dict['scanAxis']
            self.integration_time = xrd_dict['countTime']


m_package.__init_metainfo__()

# testing the parser
data_file = '/home/pepe_marquez/nomad-plugins/nomad-schema-plugin-x-ray-diffraction/tests/data/2theta-omega.xrdml'
print(parse_and_convert_file(data_file))