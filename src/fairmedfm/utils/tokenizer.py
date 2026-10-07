import clip
from transformers import AutoTokenizer
from open_clip import get_tokenizer


def tokenize_text(args, class_names):
    template = args.data_setting["text_template"]

    texts = [template.format(class_name.strip()) for class_name in class_names]

    if args.model == "BiomedCLIP":
        tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

        text_tokens = tokenizer(texts, context_length=256)
    elif args.model == "MedCLIP":
        BERT_TYPE = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(BERT_TYPE)
        tokenizer.model_max_length = 77
        text_tokens = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
    elif args.model in ["BLIP", "BLIP2"]:
        # BLIP receive raw texts as inputs
        text_tokens = texts
    elif args.model == "PLIP":
        tokenizer = AutoTokenizer.from_pretrained("vinid/plip")
        text_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    elif args.model == "SigLIP":
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        text_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    elif args.model == "MedSigLIP":
        tokenizer = AutoTokenizer.from_pretrained("google/medsiglip-448")
        text_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    elif args.model == "SigLIP2":
        tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")
        text_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    elif args.model == "CONCH":
        from conch.open_clip_custom import get_tokenizer as get_conch_tokenizer, tokenize

        text_tokens = tokenize(texts=texts, tokenizer=get_conch_tokenizer())
    else:
        text_tokens = clip.tokenize(texts, context_length=args.context_length)

    return text_tokens
