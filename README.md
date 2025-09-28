# kabu2 — ニュース駆動で日本株の有望銘柄を先回り検出（MVP）

目的: 最新ニュースを解析し、上昇が見込まれる日本株の候補銘柄を早期に抽出・スコアリングして通知します。

## 主要機能（MVP）
- RSS収集（TDnet/PR TIMESなど）→ 正規化 → ルール抽出 → スコアリング → ランキング
- 並列RSS取得・重複排除とSlack重複通知の防止
- CLIでの収集・ランキング・Slack通知に加え、履歴メンテナンス用ユーティリティを同梱
- ブラウザUIから重み調整・Slackプレビュー・CSVエクスポートが可能
- AIフィード選定は最新スコア＋履歴＋ユーザーブーストで賢く提案
- 設定(YAML)でフィード・閾値・重み付けを管理

## セットアップ
1) 依存のインストール
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) 設定ファイル
- `config.example.yaml` を参考に `config.yaml` を作成
- Slack通知を使う場合、環境変数 `SLACK_WEBHOOK_URL` を設定

## ブラウザUI
```
streamlit run kabu2/ui/app.py
```
- サイドバーで `config.yaml` / 名寄せCSV / データソース（ファイルアップロード・即時収集）を指定
- シグナル一覧でタグ頻度分析、Slackプレビュー送信、CSVダウンロードが行えます
- ワンタッチ分析タブはフィード選定・重み調整・Slack通知まで自動化し、診断テーブルで提案理由を確認できます

## 収集例（ネットワーク必要）
```
python -m kabu2 collect --config config.yaml --out data/news.jsonl --feeds tdnet prtimes
```
- `config.collectors` に追加した名前付きソースも `--feeds example_ir` のように選択できます。`type` が `rss` の場合は個別URLをfeedsと同様に扱えます。

### カスタムコレクタを追加する
`config.yaml` の `collectors` セクションに以下のようなエントリを追加すると、RSS以外の外部APIや有料フィードを統合できます。

```yaml
collectors:
  - name: premium_ir
    type: yourpkg.collectors.premium:collect
    token: YOUR_API_TOKEN
    universe: jp_smallcap
```

`type` には `モジュール:関数` 形式か `kabu2.collectors` エントリポイント名を指定します。関数は辞書設定を受け取り `NewsItem` のリストまたはイテレータを返せばOKです（非同期 `async def` もサポート）。

## ランキングと通知（CLI）
```
python -m kabu2 rank --input data/news.jsonl --top 10 --notify
```
- Slack通知は `notifier.slack.min_score` 以上・クールダウンを満たすシグナルのみを送信し、ブロック形式で投稿します

## メンテナンスコマンド
```
# JSONLから重複IDを削除（最新の発表を優先）
python -m kabu2 dedupe --config config.yaml

# 発表日時で並び替え、必要なら最新N件だけを残す
python -m kabu2 compact --config config.yaml --keep 2000

# 過去ニュースと株価でタグ重みを検証（平均リターン/回帰で提案値を出力）
python -m kabu2 backtest --config config.yaml --input data/news.jsonl --horizon 3 --output backtest_report.json
```

`backtest` サブコマンドは JSONL のヒット履歴と株価データ（Yahoo Finance）を突き合わせ、
タグごとの平均リターンや勝率を算出した上で回帰分析による重み候補を提示します。
`--write-config` オプションを指定すると、提案されたブレンド重みを YAML として書き出せます。

## ディレクトリ構成
- `kabu2/` … パッケージ本体（collector/extractor/scorer/notifier ほか）
- `data/` … ローカル保存（`news.jsonl` など）
- `config.example.yaml` … 設定テンプレート
- `requirements.txt` … 依存

## 注意
- フィードやページの利用規約遵守。商用データは各プロバイダの契約に従ってください。
- 出力は投資助言ではありません。最終判断はご自身で行ってください。

## 設定項目の意味（かんたん解説）
- weights（ルール重み）
  - ニュースのイベント種別ごとの加点/減点。例:
    - `upgrade`: 上方修正、`buyback_large`: 自社株買い(大)、`dividend_increase`: 増配、`negative_offering`: 公募増資など
  - 値が大きいほどスコアが上がり、ランキング上位に出やすくなります。負値は減点。
- thresholds（表示条件）
  - `min_score`: この値未満は非表示、`top_k`: 上位から表示する件数
- feeds（収集対象）
  - 任意のキー名 → RSS/AtomのURL。UIやCLIでキーを指定して収集可能
- collectors（プラグインコレクタ）
  - `type` で指定したコレクタ関数に設定を渡して追加データソースを呼び出します。`rss` を選べば単独RSSや有料フィードを個別キーとして扱えます。
  - `type` には `モジュール:関数` 形式か、`kabu2.collectors` エントリポイント名を指定できます（例: `yourpkg.collectors.premium:collect`）。
  - `name` を付けると CLI の `--feeds` や UI 側の表示で識別しやすくなります。
- feeds_priority（AIブースト）
  - フィードごとのブースト値。正で優先、負で抑制します（AIで選定の際に加点）。
- feeds_ai（AIチューニング）
  - `max_keys`: 提案件数、`fresh_hours`: フレッシュ判定時間、`stale_penalty`: 古いフィードの減点など、AI選定を細かく調整できます。
- storage.path（保存先）
  - 収集したニュースを追記していくJSONLファイルのパス
- notifier.slack（通知設定）
  - `enabled`: 通知のON/OFF、`webhook_env`: Webhook URLを格納する環境変数名（例: SLACK_WEBHOOK_URL）
  - `min_score`: Slack通知に載せる最小スコア
  - `cooldown_min`: 同一ニュースを再通知しない最短間隔（分）

ブラウザUIの「設定編集」タブから、これらの項目をGUIで編集・保存できます。
