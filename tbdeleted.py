import numpy as np
import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.optim import Adam
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
model = torch.nn.Linear(10, 10)
opt = Adam(model.parameters(), lr=0.0001)
torch_lr_scheduler = ExponentialLR(optimizer=opt, gamma=0.98)
# gamma ^ 10 = 10
lr1, lr2 = ExponentialLR(opt, gamma =0.98), ExponentialLR(opt, gamma=10 ** (1/10))
for i in range(20):
    if i < 10:
        lr2.step()
    else:
        lr1.step()
    print("LR:", opt.param_groups[0]['lr'],)
exit(1)

lr_values = [None] * 100
warmup_sched = StepLR()
scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                            warmup_start_value=0.00001,
                                            warmup_end_value=0.0001,
                                            warmup_duration=10,
                                            output_simulated_values=lr_values)
lr_values = np.array(lr_values)
for i in range(20):
    scheduler(opt)
    print("\n\nLR:", opt.param_groups[0]['lr'], "\n\n")

# Plot simulated values
plt.plot(lr_values[:, 0], lr_values[:, 1], label="learning rate")
plt.show()
# Attach to the trainer
