# Contributing to LeafGuard AI

Thank you for your interest in improving plant disease detection!

## Areas to Contribute

- 🌱 **New plant species** — Add to `DISEASE_DATABASE` and retrain
- 🤖 **Model improvements** — Try different architectures or training strategies  
- 🌍 **Translations** — Add multi-language support for treatment recommendations
- 📱 **Mobile app** — React Native wrapper
- 📊 **Analytics dashboard** — Prediction history and trends

## Development Setup

```bash
git clone https://github.com/yourusername/plant-disease.git
cd plant-disease

# Backend
cd backend && pip install -r requirements.txt && python main.py

# Frontend
cd frontend && npm install && npm start
```

## Pull Request Guidelines

1. Fork and create a feature branch
2. Add/update tests for your changes
3. Ensure `pytest tests/` passes
4. Ensure `npm run build` succeeds
5. Open a PR with a clear description

## Adding a New Disease

1. Add entry to `backend/utils/disease_info.py`
2. Add training images to `data/processed/train/<ClassName>/`
3. Re-run `train.py`
4. Update `NUM_CLASSES` in `model_manager.py`

## Code Style

- Python: PEP8, max 110 chars
- JavaScript: ESLint react-app config

## License

By contributing, you agree your work will be licensed under MIT.
