"""Reversible owner-scoped K1: discard only unused attention-head averaging."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import threading
from types import MethodType
from . import guards


class KineticsStateError(RuntimeError):pass
class RestorationConflict(KineticsStateError):pass
_ABSENT=object()
_OWNER=None
_OWNER_LOCK=threading.Lock()


class KineticsAdapter:
    def __init__(self,*,enabled=False,allow_unvalidated=False):
        if type(enabled) is not bool or type(allow_unvalidated) is not bool:raise TypeError('K1 flags must be bool')
        if enabled and not allow_unvalidated:raise guards.UnsupportedKinetics('K1 has no numerical acceptance; explicit allow_unvalidated=True is required')
        self.enabled,self.allow_unvalidated=enabled,allow_unvalidated
        self._active=False;self._scope=ContextVar('catpred_kinetics_scope',default=None)
        self._call_lock=threading.Lock();self._installed=[];self._guard=None
        self._reset()

    def _reset(self):
        self._record=dict(schema=1,candidate='K1_discard_unused_head_mean',enabled=self.enabled,
            allow_unvalidated=self.allow_unvalidated,engaged=False,validation_status='not_run' if self.enabled else 'off',
            optimized_calls=0,member_optimized_calls=[0]*10,member_calls=[0]*10,passthrough_calls=0,
            restored=False,active=False,failure=None,cleanup_conflicts=[],source_identity=None,
            need_weights=True,average_attn_weights=False if self.enabled else True,
            discard_unused_second_return=self.enabled,learned_values_cached=False,accepted_certificate_applies_to_K1=False)

    def summary(self):return json.loads(json.dumps(self._record))

    def _fail(self,error):
        if self._record['failure'] is None:self._record['failure']=type(error).__name__+': '+str(error)

    def _check_entries(self):
        for obj,name,old,value in self._installed:
            if obj.__dict__.get(name,_ABSENT) is not value:
                raise RestorationConflict('An installed K1 '+name+' was replaced')

    def _assign(self,obj,name,value):
        old=obj.__dict__.get(name,_ABSENT)
        # Register before mutation so partial-install exceptions are recoverable.
        self._installed.append((obj,name,old,value))
        setattr(obj,name,value)

    def _restore(self):
        conflicts=[]
        for obj,name,old,value in reversed(self._installed):
            current=obj.__dict__.get(name,_ABSENT)
            if current is old:continue  # Installation failed before assignment.
            if current is not value:
                conflicts.append(type(obj).__name__+'.'+name);continue
            if old is _ABSENT:delattr(obj,name)
            else:setattr(obj,name,old)
        self._record['cleanup_conflicts']=conflicts
        self._record['restored']=not conflicts
        self._installed=[]
        if conflicts:raise RestorationConflict('External replacements preserved: '+', '.join(conflicts))

    def _model_wrapper(self,index,original):
        def forward(_model,*args,**kwargs):
            if not self._active:raise KineticsStateError('K1 wrapper used after its activation ended')
            if threading.get_ident()!=self._thread:raise KineticsStateError('Concurrent model execution is unsupported')
            if self._scope.get() is not None:raise KineticsStateError('Nested model execution is unsupported')
            if not self._call_lock.acquire(blocking=False):raise KineticsStateError('Concurrent model execution is unsupported')
            token=None
            try:
                self._check_entries();self._guard.validate_model_call(index)
                scope=dict(index=index,attention_calls=0)
                token=self._scope.set(scope);self._record['member_calls'][index]+=1
                result=original(*args,**kwargs)
                if scope['attention_calls']!=1:raise KineticsStateError('Original model did not make exactly one eligible attention call')
                return result
            except BaseException as error:self._fail(error);raise
            finally:
                if token is not None:self._scope.reset(token)
                self._call_lock.release()
        return forward

    def _attention_wrapper(self,index,original):
        def forward(_attention,*args,**kwargs):
            scope=self._scope.get()
            if scope is None:
                self._record['passthrough_calls']+=1
                return original(*args,**kwargs)
            try:
                if not self._active or scope['index']!=index:raise KineticsStateError('Attention call belongs to a different model scope')
                if scope['attention_calls']!=0:raise KineticsStateError('Repeated MHA call within one original model forward')
                self._check_entries();self._guard.validate_attention_call(index,args,kwargs)
                # Keep the original bmm/softmax need_weights path and its q/k/v.
                # Only its unused head mean is omitted. Drop the per-head result
                # immediately so it does not remain live in MoleculeModel's `_`.
                output,unused=original(*args,need_weights=True,average_attn_weights=False)
                del unused
                scope['attention_calls']+=1
                self._record['optimized_calls']+=1;self._record['member_optimized_calls'][index]+=1
                return output,None
            except BaseException as error:self._fail(error);raise
        return forward

    @contextmanager
    def activate(self,runtime,models):
        global _OWNER
        if self._active:raise KineticsStateError('Nested activation is unsupported')
        self._reset()
        if not self.enabled:
            self._active=True;self._record['active']=True
            try:yield self
            except BaseException as error:self._fail(error);raise
            finally:self._active=False;self._record.update(active=False,restored=True)
            return
        with _OWNER_LOCK:
            if _OWNER is not None:raise KineticsStateError('Concurrent or nested K1 activation is unsupported')
            _OWNER=self
        self._thread=threading.get_ident();self._installed=[]
        primary_error=None
        try:
            self._guard=guards.build_guards(runtime,tuple(models))
            self._record['source_identity']=self._guard.source_identity
            for index,model in enumerate(self._guard.models):
                old_model,old_attention=self._guard.originals[index]
                self._assign(model,'forward',MethodType(self._model_wrapper(index,old_model),model))
                self._assign(model.multihead_attn,'forward',MethodType(self._attention_wrapper(index,old_attention),model.multihead_attn))
            self._active=True
            self._record.update(active=True,engaged=True,validation_status='experimental_unvalidated')
            yield self
        except BaseException as error:
            primary_error=error;self._fail(error);raise
        finally:
            try:
                self._restore()
            except BaseException as cleanup_error:
                self._fail(cleanup_error)
                if primary_error is None:raise
                if hasattr(primary_error,'add_note'):primary_error.add_note('K1 restoration: '+str(cleanup_error))
            finally:
                self._active=False;self._record['active']=False;self._guard=None
                with _OWNER_LOCK:
                    if _OWNER is self:_OWNER=None
