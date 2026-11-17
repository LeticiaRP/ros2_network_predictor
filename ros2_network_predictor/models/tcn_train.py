import torch
import logging
from data_pipeline import QUANTILES, HORIZONS, build_dataloader
from model_tcn import TCN, N_INPUTS, N_HORIZONS, N_QUANTILES, LossFunction

# ---------- Loss functions -------------------
BETA = 0.5 



# --------- Tranning configurations ---------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 50
N_WORKERS = 4


logger = logging.getLogger("[TCN TRAIN]")
logging.basicConfig(level=logging.DEBUG)




def train_model(train_loader, val_loader): 
    """
    Initializes the TCN model and runs the training and validation loop, saving weights with the lowest validation loss

    Arguments: 
        - train_loader: PyTorch DataLoader providing training batches 
        - val_loader: PyTorch DataLoader providing validation batches 

    Returns: 
        - TCN: The trained TCN model instance 
    """

    # Instantiate TCN model
    model = TCN(input_dim=N_INPUTS, num_horizons=N_HORIZONS, num_quantiles=N_QUANTILES).to(DEVICE)

    # Instantiate AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Custom quantile loss on the target device
    criterion = LossFunction(quantiles=QUANTILES, horizons=HORIZONS, beta=BETA).to(DEVICE)

    # Scheduler, of the val_loss does not improve in 5 epochs, the LR is cutted in half
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=5
    )

    # Tracks the lowest validation loss achivied so far for model checkpoint 
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10


    for epoch in range(EPOCHS): 

        # ------------------ TRAINING PHASE ----------------------------
        model.train()           # set model to training mode 
        train_loss = 0.0 


        for Xb, yb in train_loader: 

            # Move input windows and target horizons to GPU/CPU
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            # reset gradients, run foward pass, and compute quantile loss
            optimizer.zero_grad()
            preds = model(Xb)                   # shape: (batch, 4, 3) -> (batch, horizons, quantiles)
            loss = criterion(preds, yb)         # yb shape: (batch 4) -> (batch, horizons)  

            # backpropagate errors and update model weights
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        # compute mean training loss across all batches in the epoch
        avg_train_loss = train_loss / len(train_loader)



        # ------------------- VALIDATION PHASE ---------------------------
        model.eval()        # set model to evaluation mode

        val_loss = 0.0 


        # disable gradient tracking to reduce memory usage and speed up validation 
        with torch.no_grad(): 
            for Xb, yb, in val_loader: 
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()


        # compute mean validation loss and log epoch metrics 
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        logging.info(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # early stop and save a checkpoint whenever validation loss reaches a new minimum 
        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "saved_models/best_tcn_model.pth")
            logging.info(" -> Best model saved!")

        else: 
            patience_counter += 1 
            if patience_counter >= PATIENCE:
                logging.info(f"Early stopping activated, because the model is not improving anymore")
                break


    logging.info("Training complete!")
    return model



def main(): 

    """
    Orchestrades the end-to-end training pipeline by building and validation 
    DataLoaders and launching the TCN model training loop 

    Arguments: 
        - None 
    
    Results: 
        - None 
    """

    # path to training and validation dataset 
    TRAIN_DIR = "../../data/train"
    VAL_DIR   = "../../data/val"

    # log hardware target and initialize pytorch 
    logging.info(f"Using device: {DEVICE}")
    logging.info("1. Building DataLoaders...")

    # suffle training batches to prevent order bias; keep validation deterministic 
    train_loader = build_dataloader(TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = build_dataloader(VAL_DIR, batch_size=BATCH_SIZE, shuffle=False)

    # initialize the TCN model and run the training/validation loop 
    logging.info("2. Starting Training...")
    
    train_model(train_loader, val_loader)





    
if __name__ == '__main__': 
    main()

