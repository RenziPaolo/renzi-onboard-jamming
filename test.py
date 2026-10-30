from dataset.dataset import SARZarrDataset
from models.segmentation.msformer import MSFormer_10_384_SS
import torch


dataset = SARZarrDataset(root_dir='../../data/v1/2_processed/', paths=True, device='cuda')#, verbose=True)

dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=None,
        pin_memory=True
    )

model = MSFormer_10_384_SS(T=4, num_classes=1).to('cuda')



with torch.no_grad():
    for batch_idx, (echo, expanded_RFI, label, _) in enumerate(dataloader):#tqdm(loader, desc=f'Training Epoch {epoch}')):
        print("batch_idx:", batch_idx)
        expanded_RFI = expanded_RFI.to('cuda')
        expanded_RFI = expanded_RFI.view(-1, 2, 224, 224)
        output = model(expanded_RFI)
        print(torch.cuda.memory_summary(device=None, abbreviated=True))