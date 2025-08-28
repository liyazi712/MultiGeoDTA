import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MultiGeoDTA.dta import zinc
from MultiGeoDTA.model import DTAModel
from MultiGeoDTA.utils import Logger, Saver
from tqdm import tqdm

def _parallel_test(kwargs=None):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yp = torch.Tensor()

    with torch.cuda.device(device):
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_loader, start=1), total=len(test_loader), desc=f"Testing {midx}"):
                xd = batch['drug'].to(device)
                xp = batch['protein'].to(device)
                protein_seq = batch['full_seq'].to(device)
                pocket_seq = batch['poc_seq'].to(device)
                smile_seq = batch['smile_seq'].to(device)
                yh, protein_feats, seq_feats, compound_feats, smile_feats = model(xd, xp, protein_seq, pocket_seq, smile_seq)
                yp = torch.cat([yp, yh.detach().cpu()], dim=0)

    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_pred': yp,
    }

    return results


class DTAExperiment(object):
    def __init__(self,
                 mlp_dims=[1024, 512], mlp_dropout=0.25,
                 num_pos_emb=16, num_rbf=16,
                 contact_cutoff=8.,
                 n_ensembles=None, n_epochs=None, batch_size=None,
                 lr=0.0001, parallel=False,
                 output_dir='./output', save_log=False, save_checkpoint=False, device=None
                 ):
        self.saver = Saver(output_dir)
        self.output_dir = output_dir
        self.save_checkpoint = save_checkpoint
        self.logger = Logger(logfile=self.saver.save_dir / 'exp.log' if save_log else None)
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        print(f"device: {self.device}")

        self.dataset = zinc(
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )

        self._task_data_df_split = None
        self._task_loader = None
        self.devices = [torch.device(f'cuda:{self.device}') for _ in range(self.n_ensembles)]
        self.model_config = dict(mlp_dims=mlp_dims, mlp_dropout=mlp_dropout)
        self.build_model()
        self.criterion = F.mse_loss
        self.logger.info(self.models[0])
        self.logger.info(self.optimizers[0])

    def build_model(self):
        self.models = [DTAModel(**self.model_config).to(self.devices[i])
                       for i in range(self.n_ensembles)]
        self.optimizers = [optim.Adam(model.parameters(), lr=self.lr) for model in self.models]

    def _get_data_loader(self, dataset, shuffle=False):
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          collate_fn=dataset.collate,
                          shuffle=shuffle,
                          drop_last=False,
                          num_workers=8)

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            (split_data, split_df) = self.dataset.get_split()
            self._task_data_df_split = (split_data, split_df)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]

    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def _format_predict_df(self, results, test_df=None, esb_yp=None):
        """
        results: dict with keys y_pred, y_true
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        df['y_pred_avg'] = results['y_pred_avg']
        print(df)

        if esb_yp is not None:
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def test_unlabeled_data(self, model_file, test_df=None, save_df_name='prediction.tsv'):
        rets_list = []
        esb_yp = None
        for midx in range(self.n_ensembles):
            model = DTAModel(**self.model_config).to(self.devices[midx])
            model_path = f'./MultiGeoDTA/output/{model_file}/checkpoint_{midx + 1}.pt'
            model_state_dict = torch.load(model_path)
            model.load_state_dict(model_state_dict)
            test_loader = self.task_loader['test']
            rets = _parallel_test(
                kwargs={
                    'midx': midx + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[midx],
                },
            )
            rets_list.append(rets)
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None \
                else np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
        y_pred_avg = np.mean(esb_yp, axis=0)
        results = {
            'y_pred_avg': y_pred_avg,
        }
        results['df'] = self._format_predict_df(results, test_df=test_df, esb_yp=esb_yp)
        self.saver.save_df(results['df'], save_df_name, float_format='%g')

