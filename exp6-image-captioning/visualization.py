import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch

sns.set_style("whitegrid")
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """
    训练过程可视化工具类
    实时绘制和保存训练指标曲线
    """
    
    def __init__(self, save_dir='./visualizations', experiment_name=None):
        """
        初始化可视化器
        
        Args:
            save_dir: 可视化结果保存目录
            experiment_name: 实验名称
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # 创建实验专属目录
        self.exp_dir = self.save_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化指标存储
        self.metrics = {
            'train': {
                'epoch': [],
                'loss': [],
                'top5_acc': [],
                'batch_time': [],
                'data_time': [],
            },
            'val': {
                'epoch': [],
                'loss': [],
                'top5_acc': [],
                'bleu4': [],
            },
            'learning_rate': {
                'epoch': [],
                'encoder_lr': [],
                'decoder_lr': [],
            },
            'attention': {
                'epoch': [],
                'alpha_mean': [],
                'alpha_std': [],
                'beta_mean': [],  # 用于自适应注意力
                'beta_std': [],
            }
        }
        
        # 保存配置信息
        self.config_saved = False
    
    def save_config(self, cfg):
        """保存训练配置"""
        if not self.config_saved:
            config_path = self.exp_dir / 'config.json'
            # 转换不可序列化的对象
            cfg_serializable = {}
            for k, v in cfg.items():
                if isinstance(v, torch.device):
                    cfg_serializable[k] = str(v)
                else:
                    cfg_serializable[k] = v
            
            with open(config_path, 'w') as f:
                json.dump(cfg_serializable, f, indent=4)
            self.config_saved = True
    
    def update_train_metrics(self, epoch, loss, top5_acc, batch_time, data_time):
        """更新训练指标"""
        self.metrics['train']['epoch'].append(epoch)
        self.metrics['train']['loss'].append(loss)
        self.metrics['train']['top5_acc'].append(top5_acc)
        self.metrics['train']['batch_time'].append(batch_time)
        self.metrics['train']['data_time'].append(data_time)
    
    def update_val_metrics(self, epoch, loss, top5_acc, bleu4):
        """更新验证指标"""
        self.metrics['val']['epoch'].append(epoch)
        self.metrics['val']['loss'].append(loss)
        self.metrics['val']['top5_acc'].append(top5_acc)
        self.metrics['val']['bleu4'].append(bleu4)
    
    def update_lr_metrics(self, epoch, encoder_lr, decoder_lr):
        """更新学习率指标"""
        self.metrics['learning_rate']['epoch'].append(epoch)
        self.metrics['learning_rate']['encoder_lr'].append(encoder_lr if encoder_lr else 0)
        self.metrics['learning_rate']['decoder_lr'].append(decoder_lr)
    
    def update_attention_metrics(self, epoch, alpha_mean, alpha_std, beta_mean=None, beta_std=None):
        """更新注意力指标"""
        self.metrics['attention']['epoch'].append(epoch)
        self.metrics['attention']['alpha_mean'].append(alpha_mean)
        self.metrics['attention']['alpha_std'].append(alpha_std)
        if beta_mean is not None:
            self.metrics['attention']['beta_mean'].append(beta_mean)
            self.metrics['attention']['beta_std'].append(beta_std)
    
    def plot_loss_curves(self):
        """绘制损失曲线"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        if len(self.metrics['train']['epoch']) > 0:
            ax.plot(self.metrics['train']['epoch'], 
                   self.metrics['train']['loss'], 
                   'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        
        if len(self.metrics['val']['epoch']) > 0:
            ax.plot(self.metrics['val']['epoch'], 
                   self.metrics['val']['loss'], 
                   'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_accuracy_curves(self):
        """绘制准确率曲线"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        if len(self.metrics['train']['epoch']) > 0:
            ax.plot(self.metrics['train']['epoch'], 
                   self.metrics['train']['top5_acc'], 
                   'b-', label='Train Top-5 Acc', linewidth=2, marker='o', markersize=4)
        
        if len(self.metrics['val']['epoch']) > 0:
            ax.plot(self.metrics['val']['epoch'], 
                   self.metrics['val']['top5_acc'], 
                   'r-', label='Val Top-5 Acc', linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Top-5 Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_bleu_curve(self):
        """绘制BLEU-4曲线"""
        if len(self.metrics['val']['bleu4']) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        ax.plot(self.metrics['val']['epoch'], 
               self.metrics['val']['bleu4'], 
               'g-', label='BLEU-4', linewidth=2.5, marker='D', markersize=5)
        
        # 标注最佳BLEU分数
        best_idx = np.argmax(self.metrics['val']['bleu4'])
        best_bleu = self.metrics['val']['bleu4'][best_idx]
        best_epoch = self.metrics['val']['epoch'][best_idx]
        
        ax.plot(best_epoch, best_bleu, 'r*', markersize=15, label=f'Best: {best_bleu:.4f}')
        ax.annotate(f'Best: {best_bleu:.4f}\nEpoch: {best_epoch}',
                   xy=(best_epoch, best_bleu),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('BLEU-4 Score', fontsize=12, fontweight='bold')
        ax.set_title('Validation BLEU-4 Score', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'bleu_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_learning_rate(self):
        """绘制学习率变化曲线"""
        if len(self.metrics['learning_rate']['epoch']) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        ax.plot(self.metrics['learning_rate']['epoch'], 
               self.metrics['learning_rate']['decoder_lr'], 
               'b-', label='Decoder LR', linewidth=2, marker='o', markersize=4)
        
        if any(lr > 0 for lr in self.metrics['learning_rate']['encoder_lr']):
            ax.plot(self.metrics['learning_rate']['epoch'], 
                   self.metrics['learning_rate']['encoder_lr'], 
                   'r-', label='Encoder LR', linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_attention_statistics(self):
        """绘制注意力统计信息"""
        if len(self.metrics['attention']['epoch']) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
        
        # Alpha统计
        ax1 = axes[0]
        ax1.plot(self.metrics['attention']['epoch'], 
                self.metrics['attention']['alpha_mean'], 
                'b-', label='Mean', linewidth=2, marker='o', markersize=4)
        ax1.fill_between(self.metrics['attention']['epoch'],
                        np.array(self.metrics['attention']['alpha_mean']) - 
                        np.array(self.metrics['attention']['alpha_std']),
                        np.array(self.metrics['attention']['alpha_mean']) + 
                        np.array(self.metrics['attention']['alpha_std']),
                        alpha=0.3, label='Std Dev')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Spatial Attention (α)', fontsize=12, fontweight='bold')
        ax1.set_title('Spatial Attention Statistics', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Beta统计 (自适应注意力)
        ax2 = axes[1]
        if len(self.metrics['attention']['beta_mean']) > 0:
            ax2.plot(self.metrics['attention']['epoch'], 
                    self.metrics['attention']['beta_mean'], 
                    'r-', label='Mean', linewidth=2, marker='s', markersize=4)
            ax2.fill_between(self.metrics['attention']['epoch'],
                            np.array(self.metrics['attention']['beta_mean']) - 
                            np.array(self.metrics['attention']['beta_std']),
                            np.array(self.metrics['attention']['beta_mean']) + 
                            np.array(self.metrics['attention']['beta_std']),
                            alpha=0.3, label='Std Dev')
            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Visual Sentinel Gate (β)', fontsize=12, fontweight='bold')
            ax2.set_title('Adaptive Attention Statistics', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Adaptive Attention Data', 
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'attention_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_time_statistics(self):
        """绘制时间统计"""
        if len(self.metrics['train']['epoch']) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        ax.plot(self.metrics['train']['epoch'], 
               self.metrics['train']['batch_time'], 
               'b-', label='Batch Time', linewidth=2, marker='o', markersize=4)
        ax.plot(self.metrics['train']['epoch'], 
               self.metrics['train']['data_time'], 
               'r-', label='Data Loading Time', linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time Statistics', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'time_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_comprehensive_dashboard(self):
        """绘制综合仪表板"""
        fig = plt.figure(figsize=(20, 12), dpi=300)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss曲线
        ax1 = fig.add_subplot(gs[0, :2])
        if len(self.metrics['train']['epoch']) > 0:
            ax1.plot(self.metrics['train']['epoch'], self.metrics['train']['loss'], 
                    'b-', label='Train', linewidth=2, marker='o', markersize=3)
        if len(self.metrics['val']['epoch']) > 0:
            ax1.plot(self.metrics['val']['epoch'], self.metrics['val']['loss'], 
                    'r-', label='Val', linewidth=2, marker='s', markersize=3)
        ax1.set_title('Loss', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2 = fig.add_subplot(gs[0, 2])
        if len(self.metrics['train']['epoch']) > 0:
            ax2.plot(self.metrics['train']['epoch'], self.metrics['train']['top5_acc'], 
                    'b-', linewidth=2, marker='o', markersize=3)
        if len(self.metrics['val']['epoch']) > 0:
            ax2.plot(self.metrics['val']['epoch'], self.metrics['val']['top5_acc'], 
                    'r-', linewidth=2, marker='s', markersize=3)
        ax2.set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. BLEU-4曲线
        ax3 = fig.add_subplot(gs[1, 0])
        if len(self.metrics['val']['bleu4']) > 0:
            ax3.plot(self.metrics['val']['epoch'], self.metrics['val']['bleu4'], 
                    'g-', linewidth=2.5, marker='D', markersize=4)
            best_idx = np.argmax(self.metrics['val']['bleu4'])
            ax3.plot(self.metrics['val']['epoch'][best_idx], 
                    self.metrics['val']['bleu4'][best_idx], 
                    'r*', markersize=15)
        ax3.set_title('BLEU-4 Score', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('BLEU-4')
        ax3.grid(True, alpha=0.3)
        
        # 4. 学习率
        ax4 = fig.add_subplot(gs[1, 1])
        if len(self.metrics['learning_rate']['epoch']) > 0:
            ax4.plot(self.metrics['learning_rate']['epoch'], 
                    self.metrics['learning_rate']['decoder_lr'], 
                    'b-', linewidth=2, label='Decoder')
            if any(lr > 0 for lr in self.metrics['learning_rate']['encoder_lr']):
                ax4.plot(self.metrics['learning_rate']['epoch'], 
                        self.metrics['learning_rate']['encoder_lr'], 
                        'r-', linewidth=2, label='Encoder')
            ax4.set_yscale('log')
            ax4.legend()
        ax4.set_title('Learning Rate', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR (log scale)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Attention Alpha
        ax5 = fig.add_subplot(gs[1, 2])
        if len(self.metrics['attention']['epoch']) > 0:
            ax5.plot(self.metrics['attention']['epoch'], 
                    self.metrics['attention']['alpha_mean'], 
                    'b-', linewidth=2)
            ax5.fill_between(self.metrics['attention']['epoch'],
                            np.array(self.metrics['attention']['alpha_mean']) - 
                            np.array(self.metrics['attention']['alpha_std']),
                            np.array(self.metrics['attention']['alpha_mean']) + 
                            np.array(self.metrics['attention']['alpha_std']),
                            alpha=0.3)
        ax5.set_title('Spatial Attention (α)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Mean ± Std')
        ax5.grid(True, alpha=0.3)
        
        # 6. Attention Beta
        ax6 = fig.add_subplot(gs[2, 0])
        if len(self.metrics['attention']['beta_mean']) > 0:
            ax6.plot(self.metrics['attention']['epoch'], 
                    self.metrics['attention']['beta_mean'], 
                    'r-', linewidth=2)
            ax6.fill_between(self.metrics['attention']['epoch'],
                            np.array(self.metrics['attention']['beta_mean']) - 
                            np.array(self.metrics['attention']['beta_std']),
                            np.array(self.metrics['attention']['beta_mean']) + 
                            np.array(self.metrics['attention']['beta_std']),
                            alpha=0.3)
        ax6.set_title('Visual Sentinel (β)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Mean ± Std')
        ax6.grid(True, alpha=0.3)
        
        # 7. 时间统计
        ax7 = fig.add_subplot(gs[2, 1:])
        if len(self.metrics['train']['epoch']) > 0:
            ax7.plot(self.metrics['train']['epoch'], self.metrics['train']['batch_time'], 
                    'b-', label='Batch Time', linewidth=2, marker='o', markersize=3)
            ax7.plot(self.metrics['train']['epoch'], self.metrics['train']['data_time'], 
                    'r-', label='Data Time', linewidth=2, marker='s', markersize=3)
            ax7.legend()
        ax7.set_title('Time Statistics', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Time (s)')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Dashboard - {self.experiment_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(self.exp_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def plot_all(self):
        """绘制所有图表"""
        self.plot_loss_curves()
        self.plot_accuracy_curves()
        self.plot_bleu_curve()
        self.plot_learning_rate()
        self.plot_attention_statistics()
        self.plot_time_statistics()
        self.plot_comprehensive_dashboard()
        print(f"\n✓ All visualizations saved to: {self.exp_dir}")
    
    def save_metrics(self):
        """保存指标到JSON文件"""
        metrics_path = self.exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"✓ Metrics saved to: {metrics_path}")
    
    def generate_report(self):
        """生成训练报告"""
        report_path = self.exp_dir / 'training_report.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Training Report - {self.experiment_name}\n")
            f.write(f"{'='*60}\n\n")
            
            if len(self.metrics['val']['bleu4']) > 0:
                best_bleu_idx = np.argmax(self.metrics['val']['bleu4'])
                f.write(f"Best BLEU-4: {self.metrics['val']['bleu4'][best_bleu_idx]:.4f} ")
                f.write(f"at Epoch {self.metrics['val']['epoch'][best_bleu_idx]}\n\n")
            
            if len(self.metrics['train']['loss']) > 0:
                f.write(f"Final Train Loss: {self.metrics['train']['loss'][-1]:.4f}\n")
                f.write(f"Final Train Top-5 Acc: {self.metrics['train']['top5_acc'][-1]:.3f}%\n\n")
            
            if len(self.metrics['val']['loss']) > 0:
                f.write(f"Final Val Loss: {self.metrics['val']['loss'][-1]:.4f}\n")
                f.write(f"Final Val Top-5 Acc: {self.metrics['val']['top5_acc'][-1]:.3f}%\n\n")
            
            f.write(f"Total Epochs Trained: {len(self.metrics['train']['epoch'])}\n")
            
            if len(self.metrics['train']['batch_time']) > 0:
                avg_batch_time = np.mean(self.metrics['train']['batch_time'])
                f.write(f"Average Batch Time: {avg_batch_time:.3f}s\n")
        
        print(f"✓ Training report saved to: {report_path}")