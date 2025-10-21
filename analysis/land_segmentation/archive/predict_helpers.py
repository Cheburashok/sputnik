@torch.no_grad()
def test_folder(
    folder_path,
    model,
    preprocessing_fn,
    device="cuda" if torch.cuda.is_available() else "cpu",
    class_names=("urban_land", "agriculture_land", "forest_land"),
    show_soft=False,
    save_path=None,
):
    import matplotlib.pyplot as plt
    from pathlib import Path

    model.eval().to(device)
    folder = Path(folder_path)
    image_paths = sorted(folder.glob("*.png"))

    if len(image_paths) == 0:
        print(f"No .png images found in {folder_path}")
        return

    for img_path in image_paths:
        # --- Load image and ensure numeric dtype ---
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32)  # force numeric dtype (HxWx3)
        original_rgb = img_np.astype(np.uint8)

        # --- Pad to multiple of 32 ---
        padded_np, pad_info = _pad_to_multiple(img_np, k=32, mode="reflect")

        # --- Normalize and preprocess ---
        inp = padded_np / 255.0
        inp = preprocessing_fn(inp)  # still float32

        inp = np.transpose(inp, (2, 0, 1))  # CHW
        tensor = torch.from_numpy(inp).unsqueeze(0).to(device, dtype=torch.float32)

        # --- Predict ---
        logits = model(tensor)

        other_class_pred = logits[:, -1, :, :]
        other_ratio = other_class_pred.sum() / other_class_pred.numel()
        if other_ratio > 0.3:
            print('OTHER class ratio: ', other_ratio.item()) 
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # CxHxW float32
        probs = _crop_from_pad(probs, pad_info)

        pred_idx = np.argmax(probs, axis=0)  # HxW int
        class_maps = {}
        for cid, cname in enumerate(class_names):
            if show_soft:
                class_maps[cname] = probs[cid].astype(np.float32)
            else:
                class_maps[cname] = (pred_idx == cid).astype(np.uint8) * 255

        # --- Visualization fallback ---
        # try:
        visualize(
            original=original_rgb,
            urban_land=class_maps[class_names[0]],
            agriculture_land=class_maps[class_names[1]],
            forest_land=class_maps[class_names[2]],
        )
        # plt.figure(figsize=(10, 6))
        # plt.subplot(2, 2, 1); plt.imshow(original_rgb.astype(np.uint8)); plt.title("original"); plt.axis("off")
        # plt.subplot(2, 2, 2); plt.imshow(class_maps[class_names[0]], cmap=None if not show_soft else "viridis"); plt.title(class_names[0]); plt.axis("off")
        # plt.subplot(2, 2, 3); plt.imshow(class_maps[class_names[1]], cmap=None if not show_soft else "viridis"); plt.title(class_names[1]); plt.axis("off")
        # plt.subplot(2, 2, 4); plt.imshow(class_maps[class_names[2]], cmap=None if not show_soft else "viridis"); plt.title(class_names[2]); plt.axis("off")
        # plt.tight_layout();
        # if save_path is not None:
        #     plt.savefig(save_path, dpi=300)
        # plt.show()



@torch.no_grad()
def get_stat_timeseries(
    folder_path,
    model,
    preprocessing_fn,
    device="cuda" if torch.cuda.is_available() else "cpu",
    class_names=("urban_land", "agriculture_land", "forest_land"),
    save_path=None,
):

    model.eval().to(device)
    folder = Path(folder_path)
    image_paths = sorted(folder.glob("*.png"))
    # storage
    names = []
    rows = []  # each row is per-image class proportions (len C)

    if len(image_paths) == 0:
        print(f"No .png images found in {folder_path}")
        return
    print(len(image_paths))

    for img_path in image_paths:
        # --- Load image and ensure numeric dtype ---
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32)  # force numeric dtype (HxWx3)
        original_rgb = img_np.astype(np.uint8)

        # --- Pad to multiple of 32 ---
        padded_np, pad_info = _pad_to_multiple(img_np, k=32, mode="reflect")

        # --- Normalize and preprocess ---
        inp = padded_np / 255.0
        inp = preprocessing_fn(inp)  # still float32

        inp = np.transpose(inp, (2, 0, 1))  # CHW
        tensor = torch.from_numpy(inp).unsqueeze(0).to(device, dtype=torch.float32)

        # --- Predict ---
        logits = model(tensor)

        other_class_pred = logits[:, -1, :, :]
        other_ratio = other_class_pred.sum() / other_class_pred.numel()
        if other_ratio > 0.3:
            print('OTHER class ratio: ', other_ratio.item()) 

                # --- Spatial mean to get per-image class mix ---
        probs = torch.softmax(logits, dim=1)
        probs_c = probs.squeeze(0)  # (C, H, W)
        class_mix = probs_c.mean(dim=(1, 2))  # (C,)
        class_mix = class_mix / (class_mix.sum() + 1e-8)  # numerical safety -> sums to 1

        rows.append(class_mix.detach().cpu().numpy())
        names.append(img_path.name[:7])

        if len(rows) == 0:
            print("No images processed (possibly all skipped by OTHER threshold).")
            return
    
    series = np.vstack(rows)  # shape (N, C)
    N, C = series.shape
    if len(class_names) != C:
        print(f"Warning: class_names length ({len(class_names)}) != model channels ({C}). Using generic names.")
        class_names = tuple([f"class_{i}" for i in range(C)])

    # --- Plot stacked area (areaplot) ---
    x = np.arange(N)

    fig = plt.figure(figsize=(12, 5), dpi=120)
    plt.stackplot(x, series.T, labels=class_names)
    plt.legend(loc="upper right", ncol=min(C, 4), frameon=False)
    plt.title("Class mix over time (by filename order)")
    plt.xlabel("Image index (chronological by name)")
    plt.ylabel("Proportion (sums to 1)")
    # Thin the ticks to avoid crowding
    if N > 20:
        step = max(1, N // 20)
        plt.xticks(x[::step], [names[i] for i in range(0, N, step)], rotation=45, ha="right")
    else:
        plt.xticks(x, names, rotation=45, ha="right")
    plt.tight_layout()

    # --- Save CSV and/or figure if requested ---
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            # Treat as direct image file path
            plot_path = save_path
            csv_path = save_path.with_suffix(".csv")
        else:
            # Treat as directory
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / "class_mix_areaplot.png"
            csv_path = save_path / "class_mix_timeseries.csv"

        # Save plot
        fig.savefig(plot_path, bbox_inches="tight")
        # Save CSV
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image"] + list(class_names))
            for name, row in zip(names, series):
                writer.writerow([name] + [f"{v:.6f}" for v in row])
        print(f"Saved plot -> {plot_path}")
        print(f"Saved CSV  -> {csv_path}")
    else:
        plt.show()

    # Return data in case the caller wants to use it programmatically
    return {
        "names": names,
        "series": series,           # shape (N, C)
        "class_names": class_names  # tuple of length C
    }


def compute_class_mix_moving_averages(
    csv_path,
    window=12,
    output_file=None,
    min_periods=1,
    title=None,
):
    """
    Read the class mix time series CSV (columns: 'image', class1, class2, ...),
    compute moving averages for given windows for each class column, and save:
      - An enriched CSV with MA columns appended.
      - Per-class line plots showing original series + MAs.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV (e.g., 'class_mix_timeseries.csv').
    windows : tuple[int]
        Moving-average window sizes to compute (e.g., (12, 24)).
    save_dir : str or Path or None
        Where to save outputs. If None, uses the CSV directory.
    min_periods : int
        Minimum periods for rolling(). Defaults to 1 to avoid NaNs at the start.

    Returns
    -------
    dict
        {
          "output_csv": Path,
          "plots": {class_name: Path, ...},
          "df": pandas.DataFrame  # enriched dataframe with MA columns
        }
    """
    csv_path = Path(csv_path)

    # --- Load CSV ---
    df = pd.read_csv(csv_path)

    if "image" not in df.columns:
        raise ValueError("CSV must contain an 'image' column as the first identifier column.")

    # Identify class columns: everything except 'image' and any non-numeric
    class_cols = [c for c in df.columns if c != "image"]
    # Keep only numeric class columns
    numeric_mask = df[class_cols].apply(lambda s: pd.api.types.is_numeric_dtype(s))
    class_cols = [c for c, ok in numeric_mask.items() if ok]

    if len(class_cols) == 0:
        raise ValueError("No numeric class columns found.")

    # --- Compute moving averages for each class column ---
    ys = []
    for c in class_cols:
        ma_col = f"{c}_MA{window}"
        roll = df[c].rolling(window=window, min_periods=min_periods).mean()
        df[ma_col] = roll
        ys.append(roll)
    y = np.vstack(ys)
    
    # --- Save enriched CSV ---
    # out_csv = "class_mix_timeseries_with_MA.csv"
    # df.to_csv(out_csv, index=False)

    C, N = y.shape
    names = df.image[:N]
    print(N, C)
    x = np.arange(N)
    fig = plt.figure(figsize=(12, 5), dpi=120)
    colors = plt.cm.Set2(np.linspace(0, 1, 7))
    plt.stackplot(x, y, labels=class_names, colors=colors)

    # plt.stackplot(x, y, labels=class_names)
    # plt.legend(loc="upper right", ncol=min(C, 4), frameon=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=5)
    plt.title("Class mix over time (by filename order)" if title is None else title)
    plt.xlabel("Image index (chronological by name)")
    plt.ylabel("Proportion (sums to 1)")
    # Thin the ticks to avoid crowding
    if N > 20:
        step = max(1, N // 20)
        plt.xticks(x[::step], [names[i] for i in range(0, N, step)], rotation=45, ha="right")
    else:
        plt.xticks(x, names, rotation=45, ha="right")
    # plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    
