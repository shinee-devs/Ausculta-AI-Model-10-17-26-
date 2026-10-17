"""Optional portable model/fixture check; no private research checkout needed."""
import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
from pathlib import Path
import hashlib
import json
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[1]
def main():
    assets=ROOT/'ausculta/src/main/assets/ausculta'
    metadata=json.loads((assets/'model_metadata.json').read_text())
    path=assets/'ausculta_v143_audio.tflite'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['model_sha256']
    assert metadata['class_order']==['Asthma','Pneumonia','Healthy','Bronchitis']
    assert metadata['experimental'] and not metadata['acceptance_passed']
    model=tf.lite.Interpreter(model_path=str(path),num_threads=2,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    model.allocate_tensors();inp=model.get_input_details()[0];out=model.get_output_details()[0]
    assert inp['shape'].tolist()==[3,40000] and inp['dtype']==np.float32
    assert out['shape'].tolist()==[1,4] and out['dtype']==np.float32
    assert not any(op['op_name'].startswith('Flex') for op in model._get_ops_details())
    x=np.fromfile(assets/'golden_input.f32',dtype='<f4').reshape(3,40000)
    expected=json.loads((assets/'golden_output.json').read_text())
    model.set_tensor(inp['index'],x);model.invoke();scores=model.get_tensor(out['index'])[0]
    np.testing.assert_allclose(scores,expected['class_scores'],rtol=0,atol=1e-4)
    assert np.isfinite(scores).all() and abs(scores.sum()-1)<1e-4
    print(json.dumps(dict(conversion_fixture_passed=True,class_scores=list(map(float,scores)),experimental=True,disease_acceptance_passed=False),indent=2))
if __name__=='__main__':main()
