"""Standard-library checks of the exact discarded-weight call and torch branch."""
import ast
import hashlib
import textwrap


def require(value,message):
    if not value:raise ValueError(message)
def digest(source):return hashlib.sha256(source.encode()).hexdigest()
def tree(source):return ast.parse(textwrap.dedent(source))
def dotted(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return dotted(node.value)+'.'+node.attr
    return ''
def parents(root):return {child:node for node in ast.walk(root) for child in ast.iter_child_nodes(node)}


def prove_model_callsite(source):
    root=tree(source)
    cls=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name=='MoleculeModel')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    calls=[n for n in ast.walk(forward) if isinstance(n,ast.Call) and dotted(n.func)=='self.multihead_attn']
    require(len(calls)==1,'Expected exactly one original MHA call')
    call=calls[0];require(not call.keywords and [dotted(a) for a in call.args]==['q','k','seq_outs'],'Original MHA call arguments changed')
    owner=parents(forward)[call]
    require(isinstance(owner,ast.Assign) and len(owner.targets)==1,'Original MHA return is not a simple assignment')
    target=owner.targets[0]
    require(isinstance(target,ast.Tuple) and [dotted(v) for v in target.elts]==['seq_outs','_'],'MHA second return is not discarded')
    require(not any(isinstance(n,ast.Name) and n.id=='_' and isinstance(n.ctx,ast.Load) for n in ast.walk(forward)),'Discarded MHA weights are consumed downstream')
    return dict(source_sha256=digest(source),call_line=call.lineno,discarded_second_return=True)


def prove_torch_attention(mha_source,functional_source):
    """The flag may affect only dead mean/override forwarding in the slow path.

    Actual dispatch is additionally guarded: exact tensors, no torch-function
    override, and distinct q/k/v prohibit the native self-attention fast path.
    """
    f=tree(functional_source);p=parents(f)
    means=[n for n in ast.walk(f) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='average_attn_weights']
    require(len(means)==1,'Expected one final attention-head averaging branch')
    branch=means[0]
    expected=tree('attn_output_weights = attn_output_weights.mean(dim=1)').body
    require(not branch.orelse and [ast.dump(n,include_attributes=False) for n in branch.body]==[ast.dump(n,include_attributes=False) for n in expected],'Attention-head averaging body changed')
    for node in ast.walk(f):
        if not (isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id=='average_attn_weights'):continue
        if node is branch.test:continue
        parent=p[node]
        require(isinstance(parent,ast.keyword) and parent.arg=='average_attn_weights' and isinstance(p.get(parent),ast.Call) and dotted(p[parent].func)=='handle_torch_function','Averaging flag affects another functional operation')
    # The head mean must follow the last arithmetic assignment to attn_output
    # and remain under the original need_weights=True branch.
    ancestor=p.get(branch);found=False
    while ancestor is not None:
        if isinstance(ancestor,ast.If) and isinstance(ancestor.test,ast.Name) and ancestor.test.id=='need_weights':found=True;break
        ancestor=p.get(ancestor)
    require(found,'Head averaging is not in the original need_weights branch')
    outputs=[n for n in ast.walk(ancestor) if isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(x,ast.Name) and x.id=='attn_output' for x in ast.walk(n.targets[0] if isinstance(n,ast.Assign) else n.target))]
    require(any(n.lineno<branch.lineno for n in outputs),'Attention output is not computed before head averaging')
    # No attn_output assignment after the averaging branch may consume weights;
    # the stock optional unbatched squeeze is an output-only view.
    for n in outputs:
        if n.lineno>branch.lineno:require(not any(isinstance(x,ast.Name) and x.id=='attn_output_weights' for x in ast.walk(n.value)),'Output depends on averaged weights')
    m=tree(mha_source);mp=parents(m)
    gate=tree('query is not key or key is not value').body[0].value
    gate_nodes=[n for n in ast.walk(m) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==ast.dump(gate,include_attributes=False)]
    require(len(gate_nodes)==1,'Native self-attention identity guard changed')
    require(any(isinstance(n,ast.Assign) and any(dotted(t)=='why_not_fast_path' for t in n.targets) for n in gate_nodes[0].body),'Non-self attention no longer disables native fast path')
    allowed={'torch._native_multi_head_attention','F.multi_head_attention_forward','handle_torch_function'}
    for n in ast.walk(m):
        if not (isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='average_attn_weights'):continue
        owner=mp[n]
        if isinstance(owner,ast.keyword):owner=mp[owner]
        require(isinstance(owner,ast.Call) and dotted(owner.func) in allowed,'Averaging flag changes MHA dispatch')
    return dict(mha_forward_sha256=digest(mha_source),functional_forward_sha256=digest(functional_source),
                preserved_need_weights=True,only_dead_head_mean_removed=True,native_fast_path_excluded_by_distinct_qkv=True)
