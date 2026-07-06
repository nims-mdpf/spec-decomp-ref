import numpy as np
from typing import List

class InputController:
    """ 複数のデータにおける共有パラメータの制御を行うクラス．
        （責務）共有パラメータが明確になったパラメータ配列を
        モデル関数に入力できるパラメータ配列に変換する．
    """
    def __init__(
        self, num_input_data: int, global_idx: List[int], local_idx: List[int], 
        sg_param_size: int, bg_param_size: int) -> None:
        """
        Args:
            num_input_data (int): 統合するデータ数
            global_idx (List[int]): 共有するパラメータのidリスト
            local_idx (List[int]): 共有しないパラメータのidリスト
            sg_param_size (int): シグナルに関連するパラメータ数
            bg_param_size (int): バックグラウンドに関連するパラメータ数
        """
        self.num_input_data = num_input_data
        self.global_idx = global_idx
        self.local_idx = local_idx
        self.sg_param_size = sg_param_size
        self.bg_param_size = bg_param_size

        # リターンするパラメータ配列のサイズ
        self.output_size = (sg_param_size + bg_param_size)  
        
    def __call__(self, input_array: np.ndarray, m: int, K: int)->np.ndarray:
        """
        Args:
            input_array (np.ndarray): 共通と非共通が明確に分けられたパラメータ配列
                                cとgが共有の場合 global_idx=[0, 2], local_idx=[1, 3]
                                [
                                    ## <共有パラメータ>
                                    c_1, c_2, c_k, ... , c_K,  #共有ピークパラメータ
                                    g_1, g_2, g_k, ... , g_K,  #共有ピークパラメータ
                                    ## <個別パラメータ1>
                                    h,                         #強度パラメータ
                                    m_1, m_2, m_k, ... , m_K,  #個別ピークパラメータ
                                    g_1, g_2, g_k, ... , s_K,  #個別ピークパラメータ
                                    bg_a, bg_b                 #BGパラメータ
                                    ## <個別パラメータ1>
                                    h,                         #強度パラメータ
                                    m_1, m_2, m_k, ... , m_K,  #個別ピークパラメータ
                                    g_1, g_2, g_k, ... , s_K,  #個別ピークパラメータ
                                    bg_a, bg_b                 #BGパラメータ
                                    .....
                                ]
                                ## m -> mu, s -> sigma, g -> gamma
            m (int): モデルの番号
            K (int): 状態数

        Returns:
            np.ndarray: m番目のモデルクラスに入力できるように整形されたパラメータ配列
                                [
                                    h,                         #強度パラメータ
                                    c_1, c_2, c_k, ... , c_K,  #ピークパラメータ[比率]
                                    m_1, m_2, m_k, ... , m_K,  #ピークパラメータ[シフト]
                                    s_1, s_2, s_k, ... , g_K,  #ピークパラメータ[G幅]
                                    g_1, g_2, g_k, ... , s_K,  #ピークパラメータ[L幅]
                                    bg_a, bg_b                 #BGパラメータ
                                ]
        """

        # 出力用のパラメータ配列
        output_array = np.empty(self.output_size)
        
        # 共通パラメータ配列
        global_array = input_array[:(K*len(self.global_idx))]
        # 非共通（個別）パラメータ配列
        local_array = input_array[(K*len(self.global_idx)):]
        
        # 共有パラメータを出力パラメータに配置
        for i, idx in enumerate(self.global_idx):
            output_array[(1+idx*K):(1+(idx+1)*K)] = global_array[((i*K)):((i+1)*K)]
        
        # 強度hを先頭に配置
        each_size = m*(self.sg_param_size + self.bg_param_size - K*len(self.global_idx))
        output_array[0] = local_array[each_size]
            
        # 非共有（個別）パラメータを出力パラメータに配置
        for i, idx in enumerate(self.local_idx):
            output_array[(1+idx*K):(1+(idx+1)*K)] = local_array[(each_size+1+(i*K)):(each_size+1+((i+1)*K))]
        
        # バックグラウンドを出力パラメータに配置
        bg_idx = (each_size+self.sg_param_size-K*len(self.global_idx))
        output_array[self.sg_param_size:] = local_array[bg_idx:bg_idx+self.bg_param_size]
        
        return output_array




class Integrator:
    """複数のデータを統合するクラス．
        （責務）複数データにおけるフォワード計算と評価処理
    """
    def __init__(self, models, input_controller: InputController, K: int) -> None:
        self.input_controller = input_controller
        self.K = K # 状態数. ex) 化合物種
        self.models = models
    
    def forward(self, m: int, parameters: np.ndarray) -> np.ndarray:
        """フォワード計算

        Args:
            m (int): モデルID（データID）
            models (List[Model]): モデルクラス
            parameters (np.ndarray): 入力配列（最適化するパラメータ）

        Returns:
            np.ndarray: フィッティング関数
        """
        parameters = self.input_controller(input_array=parameters, m=m, K=self.K)
        return self.models[m].forward(parameters)
    
    
    def evaluate(self, parameters: np.ndarray) -> float:
        """評価値の計算

        Args:
            models (List[Model]): モデルクラス
            parameters (np.ndarray): 入力配列（最適化するパラメータ）

        Returns:
            float: 評価値
        """
        evaluation = 0.0
        M = len(self.models)
        for m, model in enumerate(self.models):
            param_m = self.input_controller(input_array=parameters, m=m, K=self.K)
            evaluation += model.evaluate(param_m)
        evaluation = evaluation / M
        return evaluation
