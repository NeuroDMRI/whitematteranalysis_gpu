"""Unit tests for NumPy huge-page resolution and Slicer-core hardening fallback."""
import builtins
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy

from whitematteranalysis import _memory

ROOT = Path(__file__).resolve().parents[1]
HARDEN_SCRIPT = ROOT / 'bin' / 'harden_transform_with_slicer.py'


class FakeMRMLNode:
    """Minimal MRML node test double tracking scene ownership and display nodes."""

    def __init__(self, class_name, node_id, scene, display_nodes=None):
        self.class_name = class_name
        self._id = node_id
        self._scene = scene
        self._storage = None
        self._storage_id = None
        self._transform_id = None
        self._polydata = None
        self._display_nodes = list(display_nodes or [])
        self.inverted = False

    def GetID(self):
        return self._id

    def GetScene(self):
        return self._scene

    def SetAndObserveStorageNodeID(self, storage_id):
        self._storage_id = storage_id
        self._storage = self._scene.nodes_by_id.get(storage_id)

    def GetStorageNode(self):
        return self._storage

    def CreateDefaultStorageNode(self):
        return FakeStorageNode(self.class_name + 'Storage', None)

    def SetAndObservePolyData(self, polydata):
        self._polydata = polydata

    def GetPolyData(self):
        return self._polydata

    def SetAndObserveTransformNodeID(self, transform_id):
        self._transform_id = transform_id

    def GetNumberOfDisplayNodes(self):
        return len(self._display_nodes)

    def GetNthDisplayNode(self, index):
        return self._display_nodes[index]

    def Inverse(self):
        self.inverted = True


class FakeStorageNode:
    CoordinateSystemRAS = 1

    def __init__(self, node_id, scene):
        self._id = node_id
        self._scene = scene
        self.filename = None
        self.coordinate_system = None
        self.read_calls = []
        self.write_calls = []

    def GetID(self):
        return self._id

    def GetScene(self):
        return self._scene

    def SetFileName(self, filename):
        self.filename = filename

    def SetCoordinateSystem(self, system):
        self.coordinate_system = system

    def ReadData(self, node):
        self.read_calls.append(self.filename)
        if node.class_name == 'vtkMRMLTransformNode':
            return True
        text = Path(self.filename).read_text(encoding='utf-8')
        cells = 0 if text == 'empty' else 2
        node.SetAndObservePolyData(FakePolyData(cells=cells, payload=text))
        return True

    def WriteData(self, node):
        self.write_calls.append((self.filename, self.coordinate_system))
        Path(self.filename).write_text('written:' + node.GetPolyData().payload, encoding='utf-8')
        return True


class FakeFieldData:
    def __init__(self):
        self.arrays = []

    def AddArray(self, array):
        self.arrays.append(array)


class FakePolyData:
    def __init__(self, cells=2, payload='mesh'):
        self._cells = cells
        self.payload = payload
        self._field_data = FakeFieldData()

    def GetNumberOfCells(self):
        return self._cells

    def GetFieldData(self):
        return self._field_data


class FakeDisplayNode:
    """Model display node without GetDiffusionTensorDisplayPropertiesNode."""

    def __init__(self, node_id, scene):
        self._id = node_id
        self._scene = scene

    def GetID(self):
        return self._id

    def GetScene(self):
        return self._scene


class FakeFiberDisplayNode(FakeDisplayNode):
    """Fiber bundle display node exposing GetDiffusionTensorDisplayPropertiesNode."""

    def __init__(self, node_id, scene, tensor_node):
        super().__init__(node_id, scene)
        self._tensor_node = tensor_node
        self.tensor_getter_calls = 0

    def GetDiffusionTensorDisplayPropertiesNode(self):
        self.tensor_getter_calls += 1
        return self._tensor_node


class FakeScene:
    BatchProcessState = 1

    def __init__(self):
        self._next_id = 0
        self.nodes_by_id = {}
        self.created_classes = []
        self.removed_nodes = []
        self.batch_events = []

    def _register(self, node):
        if node._id is None:
            self._next_id += 1
            node._id = 'node_%d' % self._next_id
        node._scene = self
        self.nodes_by_id[node._id] = node
        return node

    def AddNode(self, node):
        return self._register(node)

    def AddNewNodeByClass(self, class_name):
        self.created_classes.append(class_name)
        self._next_id += 1
        node_id = '%s_%d' % (class_name, self._next_id)
        if class_name == 'vtkMRMLTransformStorageNode':
            return self._register(FakeStorageNode(node_id, self))
        display_nodes = []
        if class_name == 'vtkMRMLModelNode':
            display = self._register(FakeDisplayNode(node_id + '_display', self))
            display_nodes.append(display)
        elif class_name == 'vtkMRMLFiberBundleNode':
            tensor = self._register(FakeDisplayNode(node_id + '_tensor_props', self))
            display = self._register(FakeFiberDisplayNode(node_id + '_display', self, tensor))
            display_nodes.append(display)
        return self._register(FakeMRMLNode(class_name, node_id, self, display_nodes))

    def RemoveNode(self, node):
        self.removed_nodes.append(node)
        self.nodes_by_id.pop(node.GetID(), None)
        node._scene = None

    def StartState(self, state):
        self.batch_events.append(('start', state))

    def EndState(self, state):
        self.batch_events.append(('end', state))


class FakeStringArray:
    def __init__(self):
        self.name = None
        self.values = []

    def SetName(self, name):
        self.name = name

    def InsertNextValue(self, value):
        self.values.append(value)


def _load_harden_module(search_paths=()):
    scene = FakeScene()
    fake_slicer = types.ModuleType('slicer')
    fake_slicer.mrmlScene = scene
    factory_manager = types.SimpleNamespace(searchPaths=list(search_paths))
    module_manager = types.SimpleNamespace(factoryManager=lambda: factory_manager)
    fake_slicer.app = types.SimpleNamespace(moduleManager=lambda: module_manager)

    fake_vtk = types.ModuleType('vtk')
    fake_vtk.vtkMultiThreader = types.SimpleNamespace(
        SetGlobalMaximumNumberOfThreads=Mock(),
        SetGlobalDefaultNumberOfThreads=Mock(),
    )
    fake_vtk.vtkStringArray = FakeStringArray

    with patch.dict(sys.modules, {'slicer': fake_slicer, 'vtk': fake_vtk}):
        spec = importlib.util.spec_from_file_location('test_harden_transform_with_slicer', HARDEN_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, scene


class MemoryResolveTests(unittest.TestCase):
    def test_numpy_2x_uses_private_core_without_touching_legacy_core(self):
        setter = Mock(name='numpy2_set_hugepage')

        class NumPy2Fake:
            _core = types.SimpleNamespace(
                multiarray=types.SimpleNamespace(_set_madvise_hugepage=setter)
            )

            @property
            def core(self):
                raise AssertionError('legacy numpy.core should not be touched on NumPy 2.x')

        self.assertIs(_memory._resolve_set_hugepage(NumPy2Fake()), setter)

    def test_numpy_126_stub_without_multiarray_falls_back_to_core(self):
        setter = Mock(name='numpy126_set_hugepage')
        numpy_126_stub = types.SimpleNamespace(
            _core=types.ModuleType('numpy._core'),
            core=types.SimpleNamespace(
                multiarray=types.SimpleNamespace(_set_madvise_hugepage=setter)
            ),
        )
        self.assertIs(_memory._resolve_set_hugepage(numpy_126_stub), setter)

    def test_numpy_126_stub_with_none_multiarray_falls_back_to_core(self):
        setter = Mock(name='numpy126_none_multiarray_set_hugepage')
        numpy_126_stub = types.SimpleNamespace(
            _core=types.SimpleNamespace(multiarray=None),
            core=types.SimpleNamespace(
                multiarray=types.SimpleNamespace(_set_madvise_hugepage=setter)
            ),
        )
        self.assertIs(_memory._resolve_set_hugepage(numpy_126_stub), setter)

    def test_numpy_1x_legacy_without_private_core(self):
        setter = Mock(name='numpy1x_set_hugepage')
        numpy_1x = types.SimpleNamespace(
            core=types.SimpleNamespace(
                multiarray=types.SimpleNamespace(_set_madvise_hugepage=setter)
            ),
        )
        self.assertIs(_memory._resolve_set_hugepage(numpy_1x), setter)

    def test_missing_toggle_or_multiarray_returns_none(self):
        cases = [
            ('numpy2_without_toggle', types.SimpleNamespace(
                _core=types.SimpleNamespace(multiarray=types.SimpleNamespace()),
            )),
            ('numpy1_without_toggle', types.SimpleNamespace(
                core=types.SimpleNamespace(multiarray=types.SimpleNamespace()),
            )),
            ('stub_and_core_without_multiarray', types.SimpleNamespace(
                _core=types.SimpleNamespace(),
                core=types.SimpleNamespace(),
            )),
            ('empty_module', types.SimpleNamespace()),
        ]
        for label, fake_numpy in cases:
            with self.subTest(case=label):
                self.assertIsNone(_memory._resolve_set_hugepage(fake_numpy))

    def test_real_numpy_resolution_matches_module_attribute(self):
        resolved = _memory._resolve_set_hugepage(numpy)
        self.assertIs(resolved, _memory._set_hugepage)
        self.assertTrue(resolved is None or callable(resolved))


class HardenTransformSlicerCoreFallbackTests(unittest.TestCase):
    def _make_transform_logic_module(self):
        class FakeTransformLogic:
            def __init__(self):
                self.hardened_nodes = []

            def hardenTransform(self, node):
                self.hardened_nodes.append(node.class_name)
                data = node.GetPolyData()
                node.SetAndObservePolyData(FakePolyData(cells=data.GetNumberOfCells(),
                                                        payload='hardened:' + data.payload))
                return True

        module = types.ModuleType('vtkSlicerTransformsModuleLogicPython')
        module.vtkSlicerTransformLogic = FakeTransformLogic
        return module, FakeTransformLogic

    def _make_fiber_logic_module(self):
        class FakeFiberBundleLogic:
            def __init__(self):
                self.scene = None

            def SetMRMLScene(self, scene):
                self.scene = scene

        module = types.ModuleType('vtkSlicerTractographyDisplayModuleLogicPython')
        module.vtkSlicerFiberBundleLogic = FakeFiberBundleLogic
        return module, FakeFiberBundleLogic

    def _patch_import_missing_dmri(self, transform_module):
        real_import = builtins.__import__

        def selective_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'vtkSlicerTractographyDisplayModuleLogicPython':
                raise ModuleNotFoundError("No module named 'vtkSlicerTractographyDisplayModuleLogicPython'")
            if name == 'vtkSlicerTransformsModuleLogicPython':
                return transform_module
            return real_import(name, globals, locals, fromlist, level)

        return patch.object(builtins, '__import__', side_effect=selective_import)

    def test_prepare_native_modules_falls_back_to_model_node_without_slicerdmri(self):
        extra_path = '/fake/slicer/qt-loadable-modules'
        module, _ = _load_harden_module(search_paths=[extra_path])
        transform_mod, transform_cls = self._make_transform_logic_module()
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))

        with self._patch_import_missing_dmri(transform_mod):
            fiber_logic, logic, node_class = module._prepare_native_modules()

        self.assertIsNone(fiber_logic)
        self.assertIsInstance(logic, transform_cls)
        self.assertEqual(node_class, 'vtkMRMLModelNode')
        self.assertIn(extra_path, sys.path)

    def test_prepare_native_modules_keeps_fiber_bundle_node_with_slicerdmri(self):
        module, scene = _load_harden_module()
        transform_mod, transform_cls = self._make_transform_logic_module()
        fiber_mod, fiber_cls = self._make_fiber_logic_module()

        with patch.dict(sys.modules, {
            'vtkSlicerTransformsModuleLogicPython': transform_mod,
            'vtkSlicerTractographyDisplayModuleLogicPython': fiber_mod,
        }):
            fiber_logic, logic, node_class = module._prepare_native_modules()

        self.assertIsInstance(fiber_logic, fiber_cls)
        self.assertIs(fiber_logic.scene, scene)
        self.assertIsInstance(logic, transform_cls)
        self.assertEqual(node_class, 'vtkMRMLFiberBundleNode')

    def test_main_uses_model_node_and_creates_output_dirs_without_slicerdmri(self):
        module, scene = _load_harden_module()
        transform_mod, _ = self._make_transform_logic_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            indir = root / 'inputs'
            indir.mkdir()
            (indir / 'cluster_00001.vtk').write_text('fibers1', encoding='utf-8')
            (indir / 'cluster_00002.vtp').write_text('empty', encoding='utf-8')
            tfm = root / 'transform.tfm'
            tfm.write_text('tfm', encoding='utf-8')
            outdir = root / 'nested' / 'hardened'
            splitdir = root / 'nested' / 'split'

            fake_wma = types.SimpleNamespace(io=types.SimpleNamespace(
                write_polydata=Mock(),
                read_polydata=Mock(),
            ))
            fake_sep = types.SimpleNamespace(
                read_mask_location_from_vtk=Mock(return_value=(True, numpy.array([1, 2]))),
                split_polydata=Mock(),
            )
            argv = [
                'harden_transform_with_slicer.py',
                str(indir), str(tfm), '1', str(outdir),
                '--split-output-directory', str(splitdir),
            ]
            with self._patch_import_missing_dmri(transform_mod), \
                    patch.object(module, '_prepare_split_modules', return_value=(fake_wma, fake_sep)), \
                    patch.object(sys, 'argv', argv):
                module.main()

            self.assertIn('vtkMRMLModelNode', scene.created_classes)
            self.assertNotIn('vtkMRMLFiberBundleNode', scene.created_classes)
            self.assertTrue(outdir.is_dir())
            for folder in ('tracts_left_hemisphere', 'tracts_right_hemisphere', 'tracts_commissural'):
                self.assertTrue((splitdir / folder).is_dir())
            self.assertEqual((outdir / 'cluster_00001.vtk').read_text(encoding='utf-8'),
                             'written:hardened:fibers1')
            self.assertEqual((outdir / 'cluster_00002.vtp').read_text(encoding='utf-8'), 'empty')
            self.assertEqual(fake_sep.split_polydata.call_count, 2)
            self.assertEqual(scene.nodes_by_id, {})
            self.assertEqual(scene.batch_events, [('start', scene.BatchProcessState),
                                                  ('end', scene.BatchProcessState)])

    def test_main_uses_fiber_bundle_node_when_slicerdmri_is_present(self):
        module, scene = _load_harden_module()
        transform_mod, _ = self._make_transform_logic_module()
        fiber_mod, _ = self._make_fiber_logic_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            infile = root / 'cluster_00001.vtk'
            infile.write_text('fibers', encoding='utf-8')
            tfm = root / 'transform.tfm'
            tfm.write_text('tfm', encoding='utf-8')
            outdir = root / 'out'
            argv = ['harden_transform_with_slicer.py', str(infile), str(tfm), '0', str(outdir)]
            with patch.dict(sys.modules, {
                'vtkSlicerTransformsModuleLogicPython': transform_mod,
                'vtkSlicerTractographyDisplayModuleLogicPython': fiber_mod,
            }), patch.object(sys, 'argv', argv):
                module.main()

            self.assertIn('vtkMRMLFiberBundleNode', scene.created_classes)
            self.assertNotIn('vtkMRMLModelNode', scene.created_classes)
            self.assertEqual((outdir / 'cluster_00001.vtk').read_text(encoding='utf-8'),
                             'written:hardened:fibers')
            self.assertEqual(scene.nodes_by_id, {})
            removed_ids = {node.GetID() for node in scene.removed_nodes}
            self.assertTrue(any(node_id.endswith('_tensor_props') for node_id in removed_ids))

    def test_main_accepts_legacy_two_tuple_prepare_native_modules(self):
        module, scene = _load_harden_module()
        _, transform_cls = self._make_transform_logic_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            infile = root / 'cluster.vtk'
            infile.write_text('fibers', encoding='utf-8')
            tfm = root / 'transform.tfm'
            tfm.write_text('tfm', encoding='utf-8')
            outdir = root / 'out'
            argv = ['harden_transform_with_slicer.py', str(infile), str(tfm), '0', str(outdir)]
            with patch.object(module, '_prepare_native_modules', return_value=(object(), transform_cls())), \
                    patch.object(sys, 'argv', argv):
                module.main()

            self.assertIn('vtkMRMLFiberBundleNode', scene.created_classes)
            self.assertEqual(scene.nodes_by_id, {})

    def test_remove_fiber_node_handles_model_and_fiber_display_nodes(self):
        module, scene = _load_harden_module()
        model_node = scene.AddNewNodeByClass('vtkMRMLModelNode')
        model_storage = model_node.CreateDefaultStorageNode()
        scene.AddNode(model_storage)
        model_node.SetAndObserveStorageNodeID(model_storage.GetID())
        model_node._display_nodes.append(None)

        module._remove_fiber_node(model_node)
        self.assertEqual(scene.nodes_by_id, {})

        fiber_node = scene.AddNewNodeByClass('vtkMRMLFiberBundleNode')
        fiber_storage = fiber_node.CreateDefaultStorageNode()
        scene.AddNode(fiber_storage)
        fiber_node.SetAndObserveStorageNodeID(fiber_storage.GetID())
        display = fiber_node.GetNthDisplayNode(0)

        module._remove_fiber_node(fiber_node)
        self.assertEqual(display.tensor_getter_calls, 1)
        self.assertEqual(scene.nodes_by_id, {})

    def test_harden_transform_checks_none_polydata_and_case_insensitive_vtp(self):
        module, scene = _load_harden_module()
        transform_node = scene.AddNewNodeByClass('vtkMRMLTransformNode')
        model_node = scene.AddNewNodeByClass('vtkMRMLModelNode')
        storage_node = model_node.CreateDefaultStorageNode()
        scene.AddNode(storage_node)
        model_node.SetAndObserveStorageNodeID(storage_node.GetID())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / 'out'
            splitdir = root / 'split'
            outdir.mkdir()
            splitdir.mkdir()
            infile = root / 'cluster_00001.VTP'
            infile.write_text('fibers', encoding='utf-8')

            # Case 1: ReadData succeeds but leaves polydata as None.
            bad_read_storage = Mock(wraps=storage_node)
            bad_read_storage.CoordinateSystemRAS = FakeStorageNode.CoordinateSystemRAS
            bad_read_storage.ReadData = Mock(return_value=True)
            with self.assertRaisesRegex(RuntimeError, 'empty/None'):
                module.harden_transform(
                    str(infile), transform_node, Mock(), model_node, bad_read_storage, str(outdir)
                )

            # Case 2: hardenTransform succeeds but leaves polydata as None.
            nulling_logic = types.SimpleNamespace(
                hardenTransform=lambda node: (node.SetAndObservePolyData(None), True)[1]
            )
            with self.assertRaisesRegex(RuntimeError, 'empty/None'):
                module.harden_transform(
                    str(infile), transform_node, nulling_logic, model_node, storage_node, str(outdir)
                )

            # Case 3: Uppercase .VTP with split_output_dir takes fast wma.io.write_polydata path.
            _, transform_cls = self._make_transform_logic_module()
            written = []
            fake_wma = types.SimpleNamespace(io=types.SimpleNamespace(
                write_polydata=lambda data, path: written.append((data, path)),
                read_polydata=Mock(),
            ))
            fake_sep = types.SimpleNamespace(
                read_mask_location_from_vtk=Mock(return_value=(True, numpy.array([1, 2]))),
                split_polydata=Mock(),
            )
            with patch.object(module, '_prepare_split_modules', return_value=(fake_wma, fake_sep)):
                module.harden_transform(
                    str(infile), transform_node, transform_cls(), model_node, storage_node,
                    str(outdir), str(splitdir)
                )

            self.assertEqual(len(written), 1)
            written_data, written_path = written[0]
            self.assertTrue(written_path.endswith('cluster_00001.VTP'))
            self.assertEqual(storage_node.write_calls, [])
            self.assertEqual(len(written_data.GetFieldData().arrays), 1)
            self.assertEqual(written_data.GetFieldData().arrays[0].name, 'SPACE')
            self.assertEqual(written_data.GetFieldData().arrays[0].values, ['RAS'])
            fake_sep.split_polydata.assert_called_once()


if __name__ == '__main__':
    unittest.main()
