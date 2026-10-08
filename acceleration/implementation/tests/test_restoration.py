"""Restoration contracts with lightweight sentinels; no model runtime imported."""
import importlib
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from catpred_accel.api import _Snapshot, RestorationConflict


class RestorationTests(unittest.TestCase):
    def build(self):
        def original(*args):
            return 'original'
        modules={
            'catpred.data.data': SimpleNamespace(MoleculeDataset=type('Dataset',(),{'batch_graph':original})),
            'catpred.features.featurization': SimpleNamespace(BatchMolGraph=type('Graph',(),{'__init__':original})),
            'catpred.models.model': SimpleNamespace(MoleculeModel=type('Model',(),{'forward':original})),
            'catpred.models.mpn': SimpleNamespace(MPNEncoder=type('MPN',(),{'forward':original})),
            'catpred.train.predict': SimpleNamespace(predict=original),
            'catpred.uncertainty.uncertainty_predictor': SimpleNamespace(predict=original,MVEPredictor=type('MVE',(),{'calculate_predictions':original})),
        }
        fields={
            'packing':['_COUNTS','_ARM','_INSTRUMENT','_RUST_PACK'],
            'reuse':['_CONFIG','_INSPECTION','_MODEL_IDS','_PATCHED_MODEL','_PATCHED_MPN'],
            'probes':['_CONFIG','_MODELS','_T1_APPROVED','_ORIGINAL_PREDICT','_COUNTED_PREDICT','_ORIGINAL_UNCERTAINTY_PREDICT','_PREDICT_SOURCE_SHA256','_ORIGINAL_PREDICT_MODULE','_INSTALLED'],
            'streaming':['_CONFIG','_HOST_AGGREGATE','_INSPECTION','_SOURCE_PROOF','_HOOKED_MODELS'],
        }
        components={name:ModuleType(name) for name in fields}
        for name,keys in fields.items():
            for key in keys:
                setattr(components[name],key,{})
        # A caller-owned hook must survive installation and removal.
        existing=SimpleNamespace(remove=lambda: self.fail('Caller-owned hook was removed'))
        components['probes']._MODELS[100]={'hooks':[existing]}
        components['streaming']._HOOKED_MODELS[100]=[existing]
        components['packing']._COUNTS['before']=7
        rotary=type('Rotary',(),{'rotate_queries_or_keys':original})()
        attention=type('Attention',(),{'forward':original})()
        member=SimpleNamespace(rotary_embedder=rotary,multihead_attn=attention,_forward_pre_hooks={},_forward_hooks={})
        with patch('catpred_accel.api.importlib.import_module',side_effect=lambda n:modules[n]):
            snapshot=_Snapshot(components,[member])
        return snapshot,components,member

    def alter(self,snapshot,components,member):
        def replacement(*args):
            return 'replacement'
        for obj,name,_ in snapshot.original:
            setattr(obj,name,replacement)
        member.rotary_embedder.rotate_queries_or_keys=replacement
        member.multihead_attn.forward=replacement
        removed=[]
        components['probes']._MODELS[200]={'hooks':[SimpleNamespace(remove=lambda:removed.append('probe'))]}
        components['streaming']._HOOKED_MODELS[200]=[SimpleNamespace(remove=lambda:removed.append('stream'))]
        components['packing']._COUNTS.clear()
        components['packing']._COUNTS['during']=9
        components['reuse']._CONFIG={'r1':True,'r2':True}
        return removed

    def test_functions_instance_attributes_counts_and_only_owned_hooks_restore(self):
        snapshot,components,member=self.build()
        removed=self.alter(snapshot,components,member)
        snapshot.installed_now()
        snapshot.restore()
        self.assertEqual(removed,['probe','stream'])
        self.assertEqual(components['packing']._COUNTS,{'before':7})
        self.assertEqual(components['reuse']._CONFIG,{})
        self.assertEqual(set(components['probes']._MODELS),{100})
        self.assertEqual(set(components['streaming']._HOOKED_MODELS),{100})
        self.assertNotIn('rotate_queries_or_keys',member.rotary_embedder.__dict__)
        self.assertNotIn('forward',member.multihead_attn.__dict__)
        self.assertTrue(all(getattr(obj,name) is original for obj,name,original in snapshot.original))

    def test_partial_install_failure_still_restores(self):
        snapshot,components,member=self.build()
        removed=self.alter(snapshot,components,member)
        # Installation never reached the explicit installed_now success marker.
        snapshot.restore()
        self.assertEqual(removed,['probe','stream'])
        self.assertTrue(all(getattr(obj,name) is original for obj,name,original in snapshot.original))
        self.assertEqual(components['packing']._COUNTS,{'before':7})

    def test_partial_hook_registration_without_published_handle_is_reclaimed(self):
        snapshot,components,member=self.build()
        # First hook registration succeeded; the second raised before the helper
        # could add either handle to _MODELS/_HOOKED_MODELS.
        member._forward_pre_hooks[901]=lambda *args:None
        member._forward_pre_hooks_with_kwargs={901:True}
        member._forward_hooks[902]=lambda *args:None
        member._forward_hooks_with_kwargs={902:True}
        member._forward_hooks_always_called={902:True}
        self.assertNotIn(id(member),components['probes']._MODELS)
        snapshot.restore()
        for name in ('_forward_pre_hooks','_forward_hooks','_forward_pre_hooks_with_kwargs',
                     '_forward_hooks_with_kwargs','_forward_hooks_always_called'):
            self.assertEqual(getattr(member,name),{},name)

    def test_foreign_function_replacement_is_preserved_and_reported(self):
        snapshot,components,member=self.build()
        removed=self.alter(snapshot,components,member)
        snapshot.installed_now()
        obj,name,_=snapshot.original[0]
        foreign=lambda *args:'foreign'
        setattr(obj,name,foreign)
        with self.assertRaises(RestorationConflict):
            snapshot.restore()
        self.assertIs(getattr(obj,name),foreign)
        self.assertEqual(removed,['probe','stream'])
        self.assertEqual(components['packing']._COUNTS,{'before':7})


if __name__=='__main__':
    unittest.main()
