def normalize(train, test):
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    train_scaled = (train - train_mean) / train_std
    test_scaled = (test - train_mean) / train_std

    return train_scaled, test_scaled