import re
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch 
from torch.utils.data import DataLoader
from api.API_model import AugSBERT
from pinecone import Index
import pinecone


class SearchModel():
    '''
    類似文章検索を用いた検索API用のモデル部分
    '''
    class Cleaning():
        '''
        テキストデータを前処理し、BERTに入れることができる形にする。
        '''
        #改行コードの削除
        def del_LF(self, text):
            return text.replace('\n', '').replace('\r', '')
        
        #URLタグの削除
        def del_url(self, text):
            return re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)

        #半角記号の削除
        def del_half(self, text):
            return re.sub(r'[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]', '', text)

        #全角記号の削除
        def del_mark(self, text):
            return re.sub('[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]', '', text)

        #英語の大文字表記を小文字に統一
        def change_lower(self, text):
            return text.lower()

        #各関数をまとめて行う
        def cleaning(self, text):
            text = self.del_LF(self.del_url(self.del_half(self.del_mark(self.change_lower(text)))))
            return text
    
    def __init__(self):
        #初期化
        pinecone.init(api_key='52bfa96e-aa5b-4fd7-a41b-6ca2e2bc6632', environment='us-west4-gcp-free')
        self.model = AugSBERT('sonoisa/sentence-bert-base-ja-mean-tokens-v2', 1)
        self.index = Index('recipe-search')
    
    def get_result(self, query:str, top_k:int)->list:
        '''
        クエリの情報を元にpinecone上のインデックスから検索をかける
        '''
        #クエリのベクトル化
        q_vect = self.model.encode(query).tolist()
        #pineconeで検索をかける
        result = self.index.query(q_vect, top_k=top_k, include_metadata=True)
        res_text = []
        [res_text.append({'No':n+1, 'Text':r['metadata']['sentence']}) for n, r in enumerate(result['matches'])]
        
        return res_text
    
        
    