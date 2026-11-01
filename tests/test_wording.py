import tempfile
from pathlib import Path
import unittest
import torch
from promptshield.wording import WordingBatchSampler, wording_sum


def grid(group="g"):
    return [dict(context_group=group,context_index=c,wording_index=w,label=y,
                 domain="technical" if c==0 else "general")
            for c in (0,1) for w in (0,1) for y in (0,1)]


def penalty(margins,axes="context_wording",groups=None):
    rows=grid()
    logits=torch.stack((torch.zeros_like(margins),margins),dim=1)
    return wording_sum(logits,torch.tensor([r["label"] for r in rows]),
        torch.ones(8,dtype=torch.long) if groups is None else groups,
        torch.tensor([r["context_index"] for r in rows]),torch.tensor([r["wording_index"] for r in rows]),axes)


class WordingTests(unittest.TestCase):
    def test_import_does_not_reroute_another_studys_cache(self):
        import importlib
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        import run_context_consistency as previous
        import run_wording_final as current
        before=previous.OUT
        importlib.reload(current)
        self.assertEqual(previous.OUT,before)

    def test_source_regions_exclude_navigation(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        from prepare_wording_final import SourceBody
        for cohort,attrs in (("technical",'class="sect1"'),("general",'class="article-content"'),("security",'id="Description"')):
            p=SourceBody(cohort)
            p.feed(f'<div>MENU</div><div {attrs}>real <p>article</p></div><div>FOOTER</div>')
            self.assertEqual(" ".join(p.main_text),"real  article")

    def test_family_choice_uses_all_calibration_seeds_and_prefers_control_on_tie(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        from run_wording_final import select_family
        runs={f"{a}-{s}":dict(mean_real_recall=.7,worst_real_recall=.6,threshold=.9) for a in ("context","wording") for s in (42,43,44)}
        self.assertEqual(select_family(runs)["selected_family"],"context")
        runs["wording-44"]["mean_real_recall"]=.8
        self.assertEqual(select_family(runs)["selected_family"],"wording")

    def test_sampler_coverage_replay_and_complete_grid(self):
        rows=grid("a")+grid("b")+[dict(label=i%2) for i in range(8)]
        a=list(WordingBatchSampler(rows,8,42))
        self.assertEqual(sorted(i for b in a for i in b),list(range(24)))
        self.assertEqual(a,list(WordingBatchSampler(rows,8,42)))
        self.assertTrue(all(len({rows[i].get("context_group") for i in b})==1 for b in a))
        rows[0]["wording_index"]=1
        with self.assertRaises(ValueError): WordingBatchSampler(rows,8,42)

    def test_context_control_does_not_align_wording_or_opposite_labels(self):
        # Wordings differ, contexts agree; both labels are separated by ten.
        m=torch.tensor([-5.,5.,-1.,9.,-5.,5.,-1.,9.],requires_grad=True)
        self.assertEqual(penalty(m,"context").item(),0)
        loss=penalty(m)/8
        self.assertEqual(loss.item(),8)
        loss.backward()
        updated=m.detach()-.01*m.grad
        for y in (0,1):
            torch.testing.assert_close(updated[y::2].mean(),m[y::2].mean())
        self.assertLess((updated[0]-updated[2]).abs().item(),4)

    def test_matching_edge_scale_and_accumulation_replay(self):
        m=torch.tensor([-5.,5.,-3.,7.,-3.,7.,-1.,9.],requires_grad=True)
        self.assertEqual(penalty(m,"context").item(),32)
        self.assertEqual(penalty(m,"context_wording").item(),32)
        a,b=grid("a"),grid("b")
        both=torch.cat((m,m)); logits=torch.stack((torch.zeros_like(both),both),dim=1)
        y=torch.tensor([r["label"] for r in a+b]); c=torch.tensor([r["context_index"] for r in a+b]); w=torch.tensor([r["wording_index"] for r in a+b]); g=torch.tensor([1]*8+[2]*8)
        full=wording_sum(logits,y,g,c,w,"context_wording")/16
        parts=sum(wording_sum(logits[i:i+8],y[i:i+8],g[i:i+8],c[i:i+8],w[i:i+8],"context_wording")/16 for i in (0,8))
        torch.testing.assert_close(torch.autograd.grad(full,m,retain_graph=True)[0],torch.autograd.grad(parts,m)[0])
        self.assertEqual(penalty(m,groups=torch.zeros(8,dtype=torch.long)).item(),0)

    def test_resume_weights_and_components_with_grid_and_replay(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast
        from promptshield.classifier import InjectionClassifier
        from promptshield.common import write_json,write_jsonl,sha256,read_json,read_jsonl
        from promptshield.train import train
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); backbone=root/"backbone"; splits=root/"splits"
            torch.manual_seed(3)
            DistilBertModel(DistilBertConfig(vocab_size=16,dim=16,hidden_dim=32,n_layers=1,n_heads=2,
                dropout=.1,attention_dropout=.1,max_position_embeddings=32)).save_pretrained(backbone)
            backend=Tokenizer(WordPiece({"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"benign":4,"attack":5,"other":6},unk_token="[UNK]"))
            backend.pre_tokenizer=Whitespace()
            backend.post_processor=TemplateProcessing(single="[CLS] $A [SEP]",special_tokens=[("[CLS]",2),("[SEP]",3)])
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token="[UNK]",pad_token="[PAD]",cls_token="[CLS]",sep_token="[SEP]")
            tokenizer.save_pretrained(backbone)
            rows=grid("a")+grid("b")+[dict(label=i%2) for i in range(8)]
            rows=[dict(r,id=str(i),prompt=("attack" if r["label"] else "benign")+(" other" if i%3 else "")) for i,r in enumerate(rows)]
            manifest={"splits":{}}
            for name,examples in (("train",rows),("validation",rows[:8])):
                p=splits/f"{name}.jsonl";write_jsonl(p,examples);manifest["splits"][name]={"sha256":sha256(p)}
            write_json(splits/"manifest.json",manifest)
            cfg=dict(backbone=str(backbone),data_dir=str(splits),run_dir=str(root/"whole"),seed=7,epochs=1,
                batch_size=8,accumulation_steps=2,encoder_lr=.001,head_lr=.001,weight_decay=.01,warmup_ratio=0,
                max_length=16,pooling="cls",dropout=.2,frozen=False,weighted_loss=False,max_fpr=.5,device="cpu",
                wording_batches=True,consistency_weight=.05,consistency_axes="context_wording")
            p=root/"cfg.json";write_json(p,cfg);train(p)
            expected,_,_=InjectionClassifier.load(root/"whole/last")
            cfg["run_dir"]=str(root/"resumed");write_json(p,cfg)
            (root/"resumed").mkdir();(root/"resumed/PAUSE").touch();train(p)
            self.assertEqual(read_json(root/"resumed/status.json")["status"],"paused")
            train(p,resume=True);actual,_,_=InjectionClassifier.load(root/"resumed/last")
            for a,b in zip(expected.parameters(),actual.parameters()):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            a,b=read_jsonl(root/"whole/history.jsonl")[-1],read_jsonl(root/"resumed/history.jsonl")[-1]
            for k in ("train_loss","train_ce_loss","train_consistency_loss"):self.assertAlmostEqual(a[k],b[k],places=6)


if __name__=="__main__": unittest.main()
