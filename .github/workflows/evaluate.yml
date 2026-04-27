name: Evaluate Submissions

on:
  push:
    paths:
      - "submission/**.csv"
  pull_request:
    paths:
      - "submission/**.csv"

jobs:
  evaluate:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v3
        with:
          ref: main
          token: ${{ secrets.GITHUB_TOKEN }}

      - uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: pip install pandas scikit-learn

      - name: Copy submission from PR
        if: github.event_name == 'pull_request'
        run: |
          git fetch origin pull/${{ github.event.pull_request.number }}/head:pr-branch
          git checkout pr-branch -- submission/
          git checkout main

      - name: Run evaluator
        run: python grader/evaluator.py

      - name: Update leaderboard
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/leaderboard.json submission/results.csv
          git diff --cached --quiet || git commit -m "🤖 Update leaderboard"
          git push
