"""Static rejection tests; these prove source topology, not tensor equality."""
import unittest
from catpred_kinetics.source_proof import prove_model_callsite,prove_torch_attention
MODEL='''class MoleculeModel:
 def forward(self,batch):
  seq_outs, _ = self.multihead_attn(q,k,seq_outs)
  return seq_outs
'''
MHA='''def forward(query,key,value,average_attn_weights=True):
 if query is not key or key is not value:
  why_not_fast_path = 'non-self attention'
 return F.multi_head_attention_forward(query,key,value,average_attn_weights=average_attn_weights)
'''
FUNCTIONAL='''def multi_head_attention_forward(need_weights=True,average_attn_weights=True):
 if need_weights:
  attn_output = linear(bmm(weights,v),weight,bias)
  if average_attn_weights:
   attn_output_weights = attn_output_weights.mean(dim=1)
  return attn_output,attn_output_weights
 else:
  return sdpa(q,k,v),None
'''
class SourceTests(unittest.TestCase):
 def test_discarded_model_call(self):self.assertTrue(prove_model_callsite(MODEL)['discarded_second_return'])
 def test_consumed_weights_rejected(self):
  with self.assertRaises(ValueError):prove_model_callsite(MODEL.replace('return seq_outs','return seq_outs, _'))
 def test_altered_callsite_rejected(self):
  with self.assertRaises(ValueError):prove_model_callsite(MODEL.replace('q,k,seq_outs','q,k,seq_outs,need_weights=False'))
 def test_only_dead_head_mean(self):self.assertTrue(prove_torch_attention(MHA,FUNCTIONAL)['preserved_need_weights'])
 def test_flag_changes_output_rejected(self):
  with self.assertRaises(ValueError):prove_torch_attention(MHA,FUNCTIONAL.replace('linear(bmm(weights,v),weight,bias)','linear(bmm(weights,v),weight,bias)+average_attn_weights'))
 def test_changed_reduction_rejected(self):
  with self.assertRaises(ValueError):prove_torch_attention(MHA,FUNCTIONAL.replace('mean(dim=1)','sum(dim=1)'))
 def test_native_identity_guard_required(self):
  with self.assertRaises(ValueError):prove_torch_attention(MHA.replace('query is not key or key is not value','query is key'),FUNCTIONAL)
 def test_averaged_weights_must_not_feed_output(self):
  with self.assertRaises(ValueError):prove_torch_attention(MHA,FUNCTIONAL.replace('return attn_output,attn_output_weights','attn_output = bmm(attn_output_weights,v)\n  return attn_output,attn_output_weights'))
if __name__=='__main__':unittest.main()
