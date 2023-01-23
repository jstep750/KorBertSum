"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import numpy as np

import torch
from pytorch_pretrained_bert import BertConfig


import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer
from tensorboardX import SummaryWriter
from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger
# from models.trainer import build_trainer
# build_trainer의 dependency package pyrouge.utils가 import되지 않아 직접 셀에 삽입
from others.logging import logger, init_logger
import easydict

args = easydict.EasyDict({
    "encoder":'classifier',
    "mode":'summary',
    "bert_data_path":'../bert_sample/korean',
    "model_path":'../models/bert_classifier',
    "bert_model":'../001_bert_morp_pytorch',
    "result_path":'../results/korean',
    "temp_dir":'.',
    "bert_config_path":'../001_bert_morp_pytorch/bert_config.json',
    "batch_size":1000,
    "use_interval":True,
    "hidden_size":128,
    "ff_size":512,
    "heads":4,
    "inter_layers":2,
    "rnn_size":512,
    "param_init":0,
    "param_init_glorot":True,
    "dropout":0.1,
    "optim":'adam',
    "lr":2e-3,
    "report_every":1,
    "save_checkpoint_steps":5,
    "block_trigram":True,
    "recall_eval":False,
    
    "accum_count":1,
    "world_size":1,
    "visible_gpus":'0',
    "gpu_ranks":'0',
    "log_file":'../logs/bert_classifier',
    "test_from":'../models/bert_classifier2/model_step_35000.pt'
})


def build_trainer(args, device_id, model,
                  optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    #print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        #logger.info('* number of parameters: %d' % n_params)

    return trainer

class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optim,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def summary(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        
        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls


                gold = []
                pred = []

                if (cal_lead):
                    selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                elif (cal_oracle):
                    selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                    range(batch.batch_size)]
                else:
                    sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                    # loss = self.loss(sent_scores, labels.float())
                    # loss = (loss * mask.float()).sum()
                    # batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                    # stats.update(batch_stats)

                    sent_scores = sent_scores + mask.float()
                    sent_scores = sent_scores.cpu().data.numpy()
                    selected_ids = np.argsort(-sent_scores, 1)
                # selected_ids = np.sort(selected_ids,1)
                

        return selected_ids


    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss*mask.float()).sum()
            (loss/loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)


            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        #logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

            
def summary(args, b_list, device_id, pt, step, model):
    device_id = 0
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    #print(args)

    #model.load_cp(checkpoint)
    #model.eval()

    test_iter =data_loader.Dataloader(args, _lazy_dataset_loader(b_list),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    result = trainer.summary(test_iter,step)
    return result

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

init_logger(args.log_file)
device = "cpu" if args.visible_gpus == '-1' else "cuda"
device_id = 0 if device == "cuda" else -1
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']

import argparse
import json
import os
import time
import urllib3
from glob import glob
import collections
import six
import gc
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.utils import download as _download

def do_lang ( openapi_key, text ) :
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
    requestJson = {  
        "argument": {
            "text": text,
            "analysis_code": "morp"
        }
    }
    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8", "Authorization" :  openapi_key},
        body=json.dumps(requestJson)
    )

    json_data = json.loads(response.data.decode('utf-8'))
    json_result = json_data["result"]
    
    if json_result == -1:
        json_reason = json_data["reason"]
        if "Invalid Access Key" in json_reason:
            logger.info(json_reason)
            logger.info("Please check the openapi access key.")
            sys.exit()
        return "openapi error - " + json_reason
    else:
        json_data = json.loads(response.data.decode('utf-8'))
    
        json_return_obj = json_data["return_object"]
        
        return_result = ""
        json_sentence = json_return_obj["sentence"]
        for json_morp in json_sentence:
            for morp in json_morp["morp"]:
                return_result = return_result+str(morp["lemma"])+"/"+str(morp["type"])+" "

        return return_result

def get_kobert_vocab(cachedir="./tmp/"):
    # Add BOS,EOS vocab
    tokenizer = {
        "url": "s3://skt-lsl-nlp-model/KoBERT/tokenizers/kobert_news_wiki_ko_cased-1087f8699e.spiece",
        #"fname": "/opt/ml/SumAI/bertsum/.cache/kobert_news_wiki_ko_cased-1087f8699e.spiece",
        "chksum": "ae5711deb3",
    }
    
    vocab_info = tokenizer
    vocab_file = _download(
        #vocab_info["url"], vocab_info["fname"], vocab_info["chksum"], cachedir=cachedir
        vocab_info["url"], vocab_info["chksum"], cachedir=cachedir
    )

    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
        vocab_file[0], padding_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )
    return vocab_b_obj

    
class BertData():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        #self.tokenizer = Tokenizer(vocab_file_path)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src):

        if (len(src) == 0):
            return None

        original_src_txt = [''.join(s) for s in src]


        idxs = [i for i, s in enumerate(src) if (len(s) > 0)]

        src = [src[i][:20000] for i in idxs]
        src = src[:10000]

        if (len(src) < 3):
            return None

        src_txt = [''.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = text.split(' ')
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = None
        tgt_txt = None
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt
    
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
        
class Tokenizer(object):
    
    def __init__(self, vocab_file_path):
        self.vocab_file_path = vocab_file_path
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with open(self.vocab_file_path, "r", encoding='utf-8') as reader:

            while True:
                token = convert_to_unicode(reader.readline())
                if not token:
                    break

          ### joonho.lim @ 2019-03-15
                if token.find('n_iters=') == 0 or token.find('max_length=') == 0 :

                    continue
                token = token.split('\t')[0].strip('_')

                token = token.strip()
                vocab[token] = index
                index += 1
        self.vocab = vocab
        
    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            try:
                ids.append(self.vocab[token])
            except:
                ids.append(1)
        if len(ids) > 10000:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), 10000)
            )
        return ids


def _lazy_dataset_loader(pt_file):
    dataset = pt_file    
    yield dataset
    
def News_to_input(text, openapi_key, tokenizer):
    newstemp = do_lang(openapi_key, text)
    news = newstemp.split(' ./SF ')[:-1]
    
    bertdata = BertData(tokenizer)
    sent_labels = [0] * len(news)
    tmp = bertdata.preprocess(news)
    if(not tmp): return None
    #print(tmp)
    b_data_dict = {"src":tmp[0],
               "tgt": [0],
               "labels":[0,0,0],
               "src_sent_labels":sent_labels,
               "segs":tmp[2],
               "clss":tmp[3],
               "src_txt":tmp[4],
               "tgt_txt":'hehe'}
    b_list = []
    b_list.append(b_data_dict) 
    return b_list


if __name__ == '__main__':
    
    openapi_key = '9318dc23-24ac-4b59-a99e-a29ec170bf02'
    chat_history = "뉴스 "
    user_input = '''
    일본 정부가 문재인 대통령이 도쿄올림픽을 계기로 일본을 방문하는 방향으로 한일 양국이 조율하고 있다는 15일 요미우리신문의 보도를 부인했다.
    정부 대변인인 가토 가쓰노부(加藤勝信) 관방장관은 이날 오전 정례 기자회견에서 관련 질문에 “말씀하신 보도와 같은 사실이 없는 것으로 안다”고 밝혔다.
    앞서 요미우리는 한국 측이 도쿄올림픽을 계기로 한 문 대통령의 방일을 타진했고, 일본 측은 수용하는 방향이라고 이날 보도했다.
    한국 측은 문 대통령의 방일 때 스가 요시히데(菅義偉) 총리와 처음으로 정상회담을 하겠다는 생각이라고 요미우리는 전했다.
    가토 장관은 한일 정상회담에 대한 일본 정부의 자세에 대해 “그런 사실이 없기 때문에 가정의 질문에 대해 답하는 것을 삼가겠다”고 말했다.
    그는 한국 측의 독도방어훈련에 ‘어떤 대항 조치를 생각하고 있느냐’는 질문에는 “한국 해군의 훈련에 대해 정부로서는 강한 관심을 가지고 주시하는 상황이어서 지금 시점에선 논평을 삼가겠다”고 말을 아꼈다.
    가토 장관은 “다케시마(竹島·일본이 주장하는 독도의 명칭)는 역사적 사실에 비춰봐도, 국제법상으로도 명백한 일본 고유의 영토”라며 독도 영유권 주장을 되풀이했다.
    그러면서 “다케시마 문제에 대해서는 계속 우리나라의 영토, 영해, 영공을 단호히 지키겠다는 결의로 냉정하고 의연하게 대응해갈 생각”이라고 밝혔다.
    '''
    vocab = get_kobert_vocab()
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)


    bot_input_ids = News_to_input(chat_history + user_input, openapi_key, tokenizer)
    print('------------------------------------------------------------')


    #logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load('../models/bert_classifier/model_step_1000.pt', map_location=lambda storage, loc: storage)
    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()  

    chat_history_ids = summary(args, bot_input_ids, 0, '', None, model)
    print(chat_history_ids)
    print('------------------------------------------------------------')
    print(chat_history_ids[0])
    pred_lst = list(chat_history_ids[0][:3])
    print(pred_lst)
    print('------------------------------------------------------------')
    final_text = ''
    for i,a in enumerate(user_input.split('. ')):
        if i in pred_lst:
            print(i, a)
            final_text = final_text+a+'. '
    chat_history = user_input + '<token>' +final_text
    print('------------------------------------------------------------')
    print(final_text)
    print('------------------------------------------------------------')