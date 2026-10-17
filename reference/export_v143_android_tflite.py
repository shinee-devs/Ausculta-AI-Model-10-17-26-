"""Export the selected ensemble and its full audio frontend; never train."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'experiments/audio_runtime_cache_v1'))
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_DETERMINISTIC_OPS']='1'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json
import hashlib
import joblib
import numpy as np
import tensorflow as tf
import librosa
from scipy.fft import dct

OUT=ROOT/'experiments/no_copd_v143_tflite_export_v1'
PACKAGE=ROOT/'deployment/android_v143_tflite'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

class Ensemble(tf.Module):
    def __init__(self,bundle):
        super().__init__();self.bundle=bundle
        self.nets=[p['model']._model for p in bundle['parents'][1:]]
        self.mel=tf.constant(librosa.filters.mel(sr=8000,n_fft=512,n_mels=64,fmin=0,fmax=4000,htk=False,norm='slaney').T)
        self.cosines=tf.constant(dct(np.eye(64,dtype=np.float32),type=2,axis=0,norm='ortho')[:20])
    def classify(self,features):
        probabilities=[]
        for index,parent in enumerate(self.bundle['parents']):
            scaler=parent['scaler'];x=(features-tf.constant(scaler.mean_,tf.float32))/tf.constant(scaler.scale_,tf.float32)
            model=parent['model']
            if index==0:
                heads=[]
                for head,columns in zip(model.models_,model.feature_indices_):
                    z=tf.gather(x,columns.tolist(),axis=1)
                    heads.append(tf.math.sigmoid(tf.matmul(z,tf.constant(head.coef_.T,tf.float32))+tf.constant(head.intercept_,tf.float32)))
                healthy,asthma,full,filtered=heads
                pneumonia=(1-model.filter_weight)*full+model.filter_weight*filtered
                disease=1-healthy;infectious=disease*(1-asthma)
                p=tf.concat([disease*asthma,infectious*pneumonia,healthy,infectious*(1-pneumonia)],axis=1)
            else:
                logits=self.nets[index-1](x,training=False)
                if isinstance(logits,(list,tuple)):logits=logits[0]
                p=tf.nn.softmax(logits,axis=-1)
            probabilities.append(p*self.bundle['blend_weights'][index])
        return tf.reduce_mean(tf.add_n(probabilities),axis=0,keepdims=True)
    def descriptors(self,waveforms):
        spectrum=tf.signal.stft(waveforms,frame_length=512,frame_step=128,fft_length=512,window_fn=tf.signal.hann_window,pad_end=False)
        power=tf.square(tf.abs(spectrum));mel=tf.matmul(power,self.mel)
        db=10/tf.math.log(tf.constant(10.,tf.float32))*tf.math.log(tf.maximum(mel,1e-10))
        db-=tf.reduce_max(db,axis=[1,2],keepdims=True)
        db=tf.maximum(db,-80.)
        edges=np.linspace(0,309,65,dtype=int)
        spec=tf.stack([tf.reduce_mean(db[:,a:b,:],axis=1) for a,b in zip(edges[:-1],edges[1:])],axis=2)
        mean=tf.reduce_mean(spec,axis=2);centered=spec-mean[:,:,None]
        variance=tf.reduce_mean(tf.square(centered),axis=2)
        std=tf.sqrt(variance);delta=tf.reduce_mean(tf.abs(spec[:,:,1:]-spec[:,:,:-1]),axis=2)
        mfcc=tf.matmul(self.cosines,spec)
        mfccmean=tf.reduce_mean(mfcc,axis=2)
        mfccstd=tf.sqrt(tf.reduce_mean(tf.square(mfcc-mfccmean[:,:,None]),axis=2))
        sorted_values=-tf.math.top_k(-centered,k=64,sorted=True).values
        quantiles=[]
        for q in [.1,.9]:
            position=q*63;lo=int(np.floor(position));fraction=position-lo
            quantiles.append(sorted_values[:,:,lo]*(1-fraction)+sorted_values[:,:,lo+1]*fraction)
        pieces=[mean,std,delta,mfccmean,mfccstd,*quantiles]
        for lag in [1,4,12]:pieces.append(tf.reduce_mean(centered[:,:,:-lag]*centered[:,:,lag:],axis=2)/tf.maximum(variance,1e-6))
        pieces.append(tf.reduce_mean(tf.abs(mfcc[:,:,1:]-mfcc[:,:,:-1]),axis=2))
        return tf.concat(pieces,axis=1)
    @tf.function(input_signature=[tf.TensorSpec([3,40000],tf.float32,name='waveforms')])
    def audio(self,waveforms):return {'class_scores':self.classify(self.descriptors(waveforms))}
    @tf.function(input_signature=[tf.TensorSpec([3,572],tf.float32,name='descriptors')])
    def features(self,descriptors):return {'class_scores':self.classify(descriptors)}

def main():
    assert not OUT.exists(),'Preserve any completed or failed export'
    selection=ROOT/'experiments/mobile_prototype_selection_2026_10_17.json';manifest=json.loads(selection.read_text());artifact=ROOT/manifest['artifact']
    assert sha(artifact)==manifest['artifact_sha256']
    assert manifest['class_order']==['Asthma','Pneumonia','Healthy','Bronchitis']
    bundle=joblib.load(artifact);assert bundle['blend_weights']==[.5,.25,.25]
    OUT.mkdir(parents=True);dump(OUT/'provenance.json',dict(source_artifact_sha256=sha(artifact),selection_sha256=sha(selection),plan_sha256=sha(ROOT/'experiments/no_copd_v143_tflite_export_v1_plan.json'),exporter_sha256=sha(Path(__file__)),new_fits=0,accepted=False))
    module=Ensemble(bundle)
    for name,function in [('classifier',module.features),('audio',module.audio)]:
        converter=tf.lite.TFLiteConverter.from_concrete_functions([function.get_concrete_function()],module)
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
        model=converter.convert();path=OUT/f'v143_{name}.tflite';path.write_bytes(model)
        interpreter=tf.lite.Interpreter(model_content=model,num_threads=2,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors()
        dump(OUT/f'{name}_contract.json',dict(sha256=sha(path),bytes=len(model),inputs=[dict(name=x['name'],shape=x['shape'].tolist(),dtype=str(x['dtype'])) for x in interpreter.get_input_details()],outputs=[dict(name=x['name'],shape=x['shape'].tolist(),dtype=str(x['dtype'])) for x in interpreter.get_output_details()],operators=sorted({o['op_name'] for o in interpreter._get_ops_details()})))
        print(f'Exported {name}: {len(model)}bytes',flush=True)
if __name__=='__main__':main()
