import numpy as np
import random
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import cross_encoder
from sentence_transformers import models, losses, InputExample, SentencesDataset
from sentence_transformers import cross_encoder, SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator



class Make_dataloader:
    '''
    テキストデータをデータローダ型に変換するクラス
    [コンストラクト]
    <input>
    text_data: {text1, text2}
    label: label:int or None
    <outout>
    dataloader
    '''
    def __init__(self, model):
        #用いるモデル
        self.model = model
        
    def make_example(self, data_list):
        '''
        SBERT用のDatasetを作成する
        '''
        #格納先の準備
        sample_list = []
        #データとラベルを一つずつInputExampleにして格納
        for data in data_list:
            if data['label']:
                tmp = InputExample(texts=[data['sentence1'], data['sentence2']], label=data['label']) 
            
            else:
                tmp = InputExample(texts=[data['sentence1'], data['sentence2']])
            sample_list.append(tmp)
        return sample_list
    
    def make_dataloader(self, data_list, shuffle:bool=True, batch_size:int=16):
        '''
        データローダ型にする
        '''
        #InputExampleに保存されたものをデータローダ化する
        #label_listの有無で条件分岐
        sample_list = self.make_example(data_list=data_list)
        #データローダ型に変換
        dataset=SentencesDataset(sample_list, self.model)
        dataloader=DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
        return dataloader

class AugSBERT(nn.Module):
    '''
    sentence_transformerのAPIを用いたモデル作成
    '''   
    def __init__(self, model_name:str, num_labels:int):
        #パラメータの継承
        super(AugSBERT, self).__init__()
        self.num_labels = num_labels
        #モデルの定義
        self.bert = models.Transformer(model_name)
        self.pooling = models.Pooling(self.bert.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[self.bert, self.pooling])
        
    def sampling(self, gold_data:list, target_data:list, num_epochs:int=5)->list:
        '''
        [コンストラクト]
        <param>
        ・gold_data:cross_encoderを学習させるのSilver Dataset作成時に用いる{text1, text2, label}
        ・target_data:unlabeldataでSilverDatasetの作成に用いる{text1, text2}
        <output>
        ・Silver Dataset
        
        [Sampling Strategy]
        pred_label=1は全て採用, pred_label=0はgoldと比が同じになるようにランダムに取得
        '''
        #モデルのインスタンス化
        cross_model = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        cross_model = cross_encoder.CrossEncoder(cross_model, self.num_labels)
        #モデルの学習
        cross_makeloader = Make_dataloader(cross_model)
        #評価に入れれる形にする
        dev = SentencesDataset(cross_makeloader.make_example(gold_data), cross_model)
        #DataLoader型に変換
        dataloader = cross_makeloader.make_dataloader(gold_data, shuffle=True,batch_size=16)
        #評価関数の選択
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev)
        #warmupの設定
        warmup_step = int(len(gold_data)*num_epochs*0.20)
        #モデルの学習
        cross_model.fit(
            train_dataloader=dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            warmup_steps=warmup_step,
            optimizer_params={'lr': 2e-5},
            output_path='sbert-cross-encoder'
        )
        #予測ラベルの取得
        logits = cross_model.predict(target_data, batch_size=16)
        pred = [int(logit.round(0)) for logit in logits]
        
        for n in range(len(target_data)):
            target_data[n]['label'] = pred[n]
        #labelの値ごとで分割
        tmp0 = []
        tmp1 = []
        for target in target_data:
            if target['label'] == 0:
                tmp0.append(target)
            elif target['label'] == 1:
                tmp1.append(target)
        #GoldDatasetのlabelの比を取得
        gold_label = [data['label'] for data in gold_data]
        #labelの比を取得
        gold_ratio = np.bincount(gold_label)
        pred_ratio = np.bincount(pred)
        #偏りがあるかどうかで条件分岐
        if (gold_ratio[0]/gold_ratio[1]) < (pred_ratio[0]/pred_ratio[1]):
            #Goldデータセットの日になるようにpred_label=0の数を取得
            num_pred0 = pred_ratio[1]*(gold_ratio[0]/gold_ratio[1])
            #tmp0からランダムに取得
            silver_data = random.sample(tmp0, num_pred0)
            #label=1のデータと結合する
            silver_data.extend(tmp1)
            #要素のシャッフルを行う
            silver_data = random.shuffle(silver_data)
        else:
            silver_data = target_data
        
        return silver_data
    
    def train(self, gold_data:list, target_data:list,
              loss=None, num_epochs:int=5, lr=2e-5):
        '''
        Bi-Encoder部分の訓練
        '''
        #Silver_Datasetの取得
        silver_data = self.sampling(gold_data=gold_data, target_data=target_data, num_epochs=num_epochs)
        #dataloader型に変換
        silver_loader = Make_dataloader(self.model).make_dataloader(silver_data)
        #損失の定義
        if loss is not None:
            train_loss = loss
        else:
            train_loss = losses.ContrastiveLoss(model=self.model)    
        #warm_up期間の設定
        warmup_step = int(len(silver_data)*num_epochs*0.20)
        #モデルの訓練
        self.model.fit(
            train_objectives=[(silver_loader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_step,
            optimizer_params={'lr':lr},
            output_path='./sbert-bi-encoder',
        )
        self.model.save(path='./sbert-bi-encoder/best_model')
    def encode(self, text, convert_tensor:bool=True):
        '''
        入力のベクトル化
        '''
        #モデルのロード
        train_model = self.model.load('./api/best_model')
        #ベクトル化を行う
        vect = train_model.encode(text, convert_to_tensor=convert_tensor)
        
        return vect
    