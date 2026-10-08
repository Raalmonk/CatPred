"""Framework-free metadata objects exercise actual K1 invocation guards."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from catpred_kinetics.guards import GuardSet,UnsupportedKinetics,validate_production_config

DEVICE=SimpleNamespace(type='cuda')
class Tensor:
 def __init__(self,shape=(50,10,8),dtype='float32',device=DEVICE):self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device
 def stride(self):return (self.shape[1]*self.shape[2],self.shape[2],1)
class MHA:
 def __init__(self):self.training=False;self.embed_dim=8;self.in_proj_weight=Tensor()
 def forward(self):pass
 def parameters(self):return iter(())
def functional():pass

class GuardTests(unittest.TestCase):
 def setUp(self):
  self.mha=MHA();self.model=SimpleNamespace(training=False,multihead_attn=self.mha)
  self.torch=SimpleNamespace(Tensor=Tensor,float32='float32',is_grad_enabled=lambda:False,is_autocast_enabled=lambda *a:False,
    overrides=SimpleNamespace(has_torch_function=lambda args:False),nn=SimpleNamespace(functional=SimpleNamespace(multi_head_attention_forward=functional)))
  g=GuardSet.__new__(GuardSet);g.models=[self.model];g.torch=self.torch;g.module_hooks=SimpleNamespace();g.attention_parameters=[()]
  g.mha_type=MHA;g.mha_forward=MHA.forward;g.mha_code=MHA.forward.__code__;g.functional=functional;g.functional_code=functional.__code__
  self.g=g;self.args=(Tensor(),Tensor(),Tensor());self.dispatch=patch('catpred_kinetics.guards.dispatch_clear');self.dispatch.start();self.addCleanup(self.dispatch.stop)
 def call(self,args=None,kwargs=None):return self.g.validate_attention_call(0,args or self.args,kwargs or {})
 def test_original_three_tensor_call(self):self.call()
 def test_checkpoint_training_batch32_is_not_inference_batch(self):
  model=SimpleNamespace(is_atom_bond_targets=False,loss_function='mve',args=SimpleNamespace(batch_size=32,seq_embed_dim=8,seq_self_attn_nheads=2))
  mha=SimpleNamespace(batch_first=True,_qkv_same_embed_dim=True,bias_k=None,bias_v=None,add_zero_attn=False,embed_dim=8,num_heads=2)
  validate_production_config(model,mha)
  self.assertEqual(model.args.batch_size,32)
  self.call()  # Actual inference tensors remain B=50.
 def test_gradient_enabled(self):
  self.torch.is_grad_enabled=lambda:True
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_autocast(self):
  self.torch.is_autocast_enabled=lambda *a:True
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_training(self):
  self.mha.training=True
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_other_mha_contract(self):
  with self.assertRaises(UnsupportedKinetics):self.call(kwargs={'need_weights':False})
 def test_qk_alias(self):
  with self.assertRaises(UnsupportedKinetics):self.call(args=(self.args[0],self.args[0],self.args[2]))
 def test_tensor_subclass(self):
  class Other(Tensor):pass
  with self.assertRaises(UnsupportedKinetics):self.call(args=(Other(),self.args[1],self.args[2]))
 def test_non_fp32(self):
  self.args[0].dtype='float16'
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_cpu(self):
  self.args[0].device=SimpleNamespace(type='cpu')
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_batch_change(self):
  with self.assertRaises(UnsupportedKinetics):self.call(args=(Tensor((51,10,8)),Tensor((51,10,8)),Tensor((51,10,8))))
 def test_mha_hook(self):
  self.mha._forward_hooks={1:lambda:None}
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_global_hook(self):
  self.g.module_hooks._global_forward_hooks={1:lambda:None}
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_torch_function_override(self):
  self.torch.overrides.has_torch_function=lambda args:True
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_function_replaced(self):
  self.torch.nn.functional.multi_head_attention_forward=lambda:None
  with self.assertRaises(UnsupportedKinetics):self.call()
 def test_precision_settings_drift(self):
  self.g.runtime=SimpleNamespace(_active=True,_manifest={'engaged':True})
  self.g.settings={'precision':'before'};self.g.identity=SimpleNamespace(runtime_settings=lambda t:{'precision':'changed'})
  with self.assertRaisesRegex(UnsupportedKinetics,'Precision/thread'):self.g.validate_runtime()
if __name__=='__main__':unittest.main()
