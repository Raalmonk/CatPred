import gzip,importlib.util,io,json,math,tempfile,unittest
from pathlib import Path

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('old_analysis',Path(__file__).resolve().parents[1]/'history/analyze_kinetics.whole_json.py')
new=module('stream_analysis',Path(__file__).resolve().parents[1]/'analyze_kinetics.py')

class StreamingTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.chunk=new.STREAM_CHUNK_CHARS;new.STREAM_CHUNK_CHARS=11
 def tearDown(self):new.STREAM_CHUNK_CHARS=self.chunk;self.temp.cleanup()
 def check(self,events,metadata=None,compressed=False,metadata_first=False):
  trace={'traceEvents':events}
  if metadata_first:trace=dict(metadata or {},**trace)
  else:trace.update(metadata or {})
  p=self.root/('trace.json.gz' if compressed else 'trace.json')
  payload=json.dumps(trace,ensure_ascii=False,separators=(',',':'))
  if compressed:
   with gzip.open(p,'wt',encoding='utf-8') as f:f.write(payload)
  else:p.write_text(payload)
  self.assertEqual(old.trace_analysis(p),new.trace_analysis(p))
  reader=new._TraceJSONReader(io.StringIO(payload),chunk_chars=1,max_value_chars=1024)
  self.assertEqual(list(reader.events()),events);self.assertTrue(reader.complete)
  self.assertLessEqual(reader.peak_buffer_chars,1024)
  self.assertEqual(reader.device_properties,trace.get('deviceProperties'))
 def mean(self,ext=1,dims=None,**args):
  return {'cat':'cpu_op','ph':'X','name':'aten::mean','dur':3.25,'args':dict({'External id':ext,'Input Dims':dims},**args)}
 def kernel(self,ext=1,dur=1.25):return {'cat':'kernel','ph':'X','name':'kernel','dur':dur,'args':{'External id':ext}}
 def test_empty(self):self.check([])
 def test_events_before_and_after(self):
  self.check([self.kernel(1,1e16),self.mean(2,[[50,6,128,128]]),self.kernel(2,1.),self.mean(1,[[50,6,128,128]]),self.kernel(1,-1e16),self.kernel(2,1.)],{'deviceProperties':[{'name':'GPU 🚀','compute':12,'other':True}]})
 def test_original_float_addition_order(self):
  events=[self.kernel(1,1e16),self.kernel(2,1.),self.kernel(1,-1e16),self.kernel(2,1.),self.mean(1,[[1,2,3,3]]),self.mean(2,[[1,2,3,3]])]
  self.check(events);self.assertEqual(new.trace_analysis(self.root/'trace.json')['directly_linked_mean_kernel_duration_us'],sum([1e16,1.,-1e16,1.]))
 def test_known_ids_no_replay(self):self.check([self.mean(),self.kernel(),self.kernel(2)],{'deviceProperties':None},compressed=True,metadata_first=True)
 def test_missing_shape(self):self.check([{'cat':'cpu_op','ph':'X','name':'aten::mean','args':{'External id':1}},self.kernel()])
 def test_shape_fallback_and_examples(self):
  events=[{'cat':'cpu_op','ph':'X','name':'aten::mean','dur':x/10,'args':{'External id':7,'Input Shapes':[[50,6,1024,1024],[]],'Concrete Inputs':[str(x),'[1]']}} for x in range(5)]
  self.check(events+[self.kernel(7)])
 def test_cpu_and_backend_counts(self):self.check([{'cat':'cpu_op','ph':'X','name':n,'args':{'Input Dims':[]}} for n in ['aten::bmm','aten::_softmax','aten::native_multi_head_attention','aten::scaled_dot_product_attention']]+[{'cat':'kernel','ph':'X','name':'Flash_attention'}])
 def test_string_external_ids_and_missing_id(self):self.check([self.kernel('x'),self.mean('x'),self.mean(None),{'cat':'kernel','name':'k'}])
 def test_top_level_metadata(self):self.check([self.mean()],{'nested':{'x':[True,False,None,-1.2e30,'escaped " ] } \\']},'deviceProperties':{'name':'猫','c':1.23e-20},'at_end':None},metadata_first=True)
 def test_operators_file_unchanged(self):
  (self.root/'trace_operators.json').write_text(json.dumps([{'operator':'aten::mean','self_device_time_us':3.25},{'operator':'kernel','self_device_time_us':1.},{'operator':'aten::other','self_device_time_us':2.}]))
  self.check([self.mean()])
 def test_malformed_and_incomplete(self):
  bad=['','{}','{"traceEvents":','{"traceEvents": [','{"traceEvents": [{','{"traceEvents": [1]}','{"traceEvents": [true]}','{"traceEvents": [{"cat":"cpu_op",}]}','{"traceEvents": [],}','{"traceEvents": [], "x":1e}','{"traceEvents": []} garbage','{"traceEvents": []}{}','{"traceEvents": [{"name":"unterminated}]}','{"traceEvents": [false,]}']
  for text in bad:
   with self.subTest(text=text):
    reader=new._TraceJSONReader(io.StringIO(text),chunk_chars=3,max_value_chars=256)
    with self.assertRaises((ValueError,KeyError,TypeError)):list(reader.events())
    self.assertFalse(reader.complete)
 def test_duplicate_top_level_rejected(self):
  with self.assertRaisesRegex(ValueError,'Duplicate top-level'):list(new._TraceJSONReader(io.StringIO('{"traceEvents":[],"traceEvents":[]}')).events())
 def test_bounded_value_explicit_rejection(self):
  reader=new._TraceJSONReader(io.StringIO(json.dumps({'traceEvents':[{'name':'x'*200}]})),chunk_chars=7,max_value_chars=64)
  with self.assertRaisesRegex(ValueError,'bounded buffer'):list(reader.events())
  self.assertLessEqual(reader.peak_buffer_chars,64)
 def test_truncated_gzip_rejected(self):
  p=self.root/'broken.json.gz';p.write_bytes(gzip.compress(b'{"traceEvents": []}')[:-4])
  with self.assertRaises((EOFError,OSError,ValueError)):new.trace_analysis(p)
 def test_irrelevant_metadata_not_retained(self):
  text=json.dumps({'metadata':['irrelevant']*20,'deviceProperties':[{'x':1}], 'traceEvents':[self.kernel()]})
  reader=new._TraceJSONReader(io.StringIO(text),chunk_chars=7,max_value_chars=1024)
  self.assertEqual(len(list(reader.events())),1)
  self.assertNotIn('metadata',vars(reader));self.assertEqual(reader.device_properties,[{'x':1}])

if __name__=='__main__':unittest.main()
