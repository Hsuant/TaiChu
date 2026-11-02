"""Compatibility-free launcher for the canonical pretraining entrypoint."""

from trainer.train_pretrain import main as train_pretrain


def main():
    train_pretrain()


if __name__ == "__main__":
    main()
