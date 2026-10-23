import pickle
from pathlib import Path
import tempfile
import unittest

import matplotlib as mpl

from phasis.plot_style import portable_plot_fonts
from phasis.stages import locus_plots, output


class PortablePlotFontTests(unittest.TestCase):
    def test_restores_settings_after_render_error(self):
        @portable_plot_fonts
        def failing_render():
            self.assertEqual(mpl.rcParams['font.family'], ['DejaVu Sans'])
            self.assertEqual(mpl.rcParams['font.weight'], 'normal')
            raise RuntimeError('test render failure')

        with mpl.rc_context({'font.family': 'PhasisMissingTestFont', 'font.weight': 'semibold'}):
            with self.assertRaisesRegex(RuntimeError, 'test render failure'):
                failing_render()
            self.assertEqual(mpl.rcParams['font.family'], ['PhasisMissingTestFont'])
            self.assertEqual(mpl.rcParams['font.weight'], 'semibold')

    def test_decorated_renderers_remain_picklable_for_spawn_workers(self):
        for render in (locus_plots._write_single_locus_plot, locus_plots._write_placeholder_plot,
                       output.plot_report_heat_map, output.plot_phasAbundance_heat_map,
                       output.plot_totalAbundance_heat_map, output.plot_howell_score_heat_maps):
            with self.subTest(render=render.__name__):
                self.assertIs(pickle.loads(pickle.dumps(render)), render)

    def test_placeholder_paths_use_portable_fonts(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mpl.rc_context({'font.family': 'PhasisMissingTestFont', 'font.weight': 'semibold',
                             'text.usetex': True}), \
             self.assertNoLogs('matplotlib', level='WARNING'):
            locus_path = Path(tmp)/'locus.pdf'
            heatmap_path = Path(tmp)/'heatmap.pdf'
            locus_plots._write_placeholder_plot(str(locus_path), 'Test locus', 'No data')
            output._empty_plot_placeholder(str(heatmap_path), 'No PHAS calls')
            for path in (locus_path, heatmap_path):
                self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
