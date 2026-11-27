# taskalign

**「視覚言語モデルにおけるタスク類似性に基づく内部表現整合学習に関する検討」の実装．**（`taskalign` はリポジトリ名でありパッケージ名で，題目の「タスク類似性に基づく内部表現整合」を縮めたもの．手法の名前としては使わない．2026-11-15 に `cotar` から改名した．）

GQA の各質問には functional program が付いている．その**演算子列**（例 `select > relate > query`）を**タスク署名**と呼び，署名の一致をもって**タスク類似性**とする．署名が一致するタスクどうしの中間層表現を supervised contrastive loss で引き寄せる——これが**内部表現整合**である．言語モデル損失への補助項として足すだけで，掴むのは生の hidden state そのもの．

問いと結果は研究文書にある．**この README は「どう動かすか」だけを扱う．**

| どこに | 何が |
| --- | --- |
| [.claude/context.md](.claude/context.md) | **研究文書**．問い・手法・検証方法・結果・答え・次の一手．これ1本で研究が分かる |
| [.claude/context-philosophy.md](.claude/context-philosophy.md) | 上を**どう書くか**の原則．研究の中身には依存しない |
| [references/](references/) | 論文の要約．1本につき1ファイル．研究文書が引く論文と，引いていないが関連する論文の両方がある．目録は [references/README.md](references/README.md)，要約を作っていない文献の調べは `references/novelty-check.md` |
| [papers/](papers/) | 論文の原稿（学会予稿・修士論文）．組版の決め事は [papers/principles.md](papers/principles.md)，予稿の書式検査は `papers/pcsj-imps-2026/check.py` |
| [presentations/](presentations/) | 発表の成果物（ゼミ・学会）と，その作り方の原則（`principles.md`）．成果物は作った時点の記録であって，最新の結果ではない．学会発表の手引き（`…_設計と実装.md`・`…_想定問答.md`）は，処理の中身と想定される質問への答えを口頭で説明できる形にまとめたもの．`…_質疑.md` は，当日に受けた質問を発表者の聞き取りのまま残し，研究文書で測ったことと測っていないことを添えたもの |
| [snapshots/](snapshots/) | 各 run が残した記録．重みを除いた写しなので commit できる |
| [analyses/](analyses/) | run をまたいで測った結果．研究文書の数値の出どころ |

## 準備

PyTorch は GPU の世代ごとに index が違うので**先に**入れる．

```bash
conda create -n taskalign -c conda-forge python=3.12 && conda activate taskalign
conda env config vars set PYTHONDONTWRITEBYTECODE=1 && conda activate taskalign   # __pycache__ を作らない
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e .
```

`pip install -e .` が `taskalign` をパスに通すので，`scripts/` のどれもが作業ディレクトリによらず `taskalign` を import できる．SmolVLM-500M の重み（約1GB）は，最初にモデルを読むスクリプトが Hugging Face から `~/.cache/huggingface/` に取ってくる．lint の ruff はこの環境に入らない（`pip install ruff`）．

**flash-attn は任意（Linux のみ）**——無ければ自動で SDPA に落ちる．ただし報告した run は flash-attn 2.8.3.post1 で回してある（各 run の `log.txt` の環境バナー）ので，学習を足す機には入れる．配布されている wheel が無ければソースからビルドすることになり，それには次の3つが要る（2026-11-25 に Ubuntu 26.04・RTX 5090 の機で確かめた）．

- **torch を 2.13.0 に留める．** flash-attn 2.8.3.post2 は C++17 でビルドするが，torch 2.15.0 のヘッダは C++20 を求めるので通らない．2.13.0 は報告した run の版でもある．上の3行目を `pip install torch==2.13.0 torchvision==0.28.0 --index-url …` に替える．
- **CUDA のコンパイラを conda-forge だけから入れる．** `--override-channels` を外すと既定のチャンネルが混ざり，conda は環境から python と pip を取り除く．
- **ヘッダの場所を教える．** conda-forge の `cuda-toolkit` はヘッダを `targets/x86_64-linux/` の下に置くので，`CUDA_HOME` だけでは `cuda_runtime.h` が見つからない．

```bash
conda install -c conda-forge --override-channels cuda-toolkit=13.2
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib
export FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 NVCC_THREADS=2   # 120 は RTX 50 系．並列数はこの機（RAM 32GB）で通った値
pip install ninja
pip install flash-attn --no-build-isolation                 # この機で十数分
```

SDPA と flash-attn とで，`probe_dependence_map.py reported --prompts 8` の表の値は 0.001 以内で一致した（同日．torch 2.15.0 + SDPA と，torch 2.13.0 + flash-attn 2.8.3.post2．transformers はどちらも 5.24.1）．学習ループの [train4all](https://github.com/tomoking2004/train4all) は**リリースタグに固定**してある（どのタグかは `pyproject.toml`）——`main` を追うと，指標の並び順のような出力の形が実験の途中で変わりうる．

データセットは [GQA 公式](https://cs.stanford.edu/people/dorarad/gqa/download.html) から取り，`cfg.datasets_root/gqa/` に置く（`images/`・`questions/`・`eval/`）．`train_balanced_questions.json` は 813MB あるので，これを読むスクリプトは立ち上がりに数分かかる．

**パスの起点は2つある．** リポジトリが持つのはコード・文書・snapshots・analyses——版管理に載せる物である．`cfg.work_root`（既定は `~/TMU/Master/study/taskalign`）が持つのはデータセットと `runs/`——数十GBあって，版管理にも同期フォルダにも置けない物である．どちらがどちらかは `taskalign/config.py` が答える．

## 走らせる

```bash
bash scripts/run_plan.sh                   # 研究文書 §8 の次の測定を一つのジョブで（後述．GPU・約28時間）
python scripts/train.py                    # 3群 × 乱数種 45・46 を回す（研究文書 §8.1．GPU・約17時間半）
python scripts/sweep.py strength           # 整合の強さを振る．layer で層，position で位置（GPU・1 run あたり約3時間）
python scripts/generate.py                 # モデルが読み込めて受け答えするかだけを見る
python scripts/audit_dataset.py            # データ側の数値（分割・署名分布・含意対・答えの語彙）
python scripts/summarize_runs.py           # run 側の数値（表現の幾何・損失・正解率・対にした差と区間）
python scripts/summarize_sweep.py          # 読み取り精度と正解率を，振った設定ごとに並べる
python scripts/probe_signature.py          # 署名が線形に読めるか
python scripts/probe_operators.py          # プローブが見ていない演算子の組み合わせが読めるか
python scripts/stratify_by_frequency.py    # 正解率を署名の頻度で層別し直す
python scripts/probe_bypass.py             # 整合した構造が出力の読む方向に載っているか（§A.4 第1段）
python scripts/probe_bypass_pullback.py    # その方向を第16層へ引き戻して問い直す（§A.4 第2段）
python scripts/probe_dependence.py         # 答えが第16層にどれだけ頼っているか（§A.4 第3段）
python scripts/probe_dependence_map.py reported   # それを全層・全位置で測り，次に整合を掛ける場所を選ぶ（§8.5 第1歩）
python scripts/summarize_relocation.py     # 選んだ場所に掛けた run を読み，§8.5 第2歩の表の行を決める
python scripts/probe_drift.py              # 画像を加工したとき表現がどれだけ動くか（§A.6）
python scripts/normalize_drift.py          # その距離を，対照の分布の形と別の比の作り方で読み直す（§A.6）
python scripts/update_poster.py            # その結果をポスターの3.3節へ書く（§8.1 の読み．torch 不要）
python scripts/update_progress.py          # analyses/ にある結果から 11/27 の進捗報告を組む（torch 不要）
python scripts/check_tidy.py               # tmp/ の形と __pycache__ を確かめる（Stop フックも走らせる）
```

**モデルを動かすのは7本**——`train.py`・`sweep.py`・`generate.py` と，`probe_bypass_pullback.py`・`probe_dependence.py`・`probe_dependence_map.py`（ヤコビ積を取るのでモデルを走らせる），`probe_drift.py`（加工した画像でモデルを走らせ直す）である．残る12本はモデルを走らせない．`probe_bypass.py` を除いて重みも要らないので，GPU の無い機でも回せる．ただし6本（`audit_dataset.py`・`probe_signature.py`・`probe_operators.py`・`stratify_by_frequency.py`・`summarize_sweep.py`・`probe_bypass.py`）は GQA の質問ファイルを読むので，それを置いた機で回す．分析の13本（`train.py`・`sweep.py`・`generate.py`・`update_poster.py`・`update_progress.py`・`check_tidy.py` を除いたもの）は，それぞれ自分と同じ名前の JSON を `analyses/` に書く（`probe_signature.py` → `analyses/probe-signature.json`．`probe_drift.py` は問ごとの距離を同じ名前の `.pt` にも書く）．どの実験を読むかは `taskalign/analysis/experiment.py` の `TIMESTAMP` が決める（§8.1 で足した乱数種 45・46 のバッチは `ADDED_TIMESTAMP`）．`sweep.py` が学習した run（層・強さ・位置を変えたもの）は，timestamp を書き写さなくても，分析の側が variant の名前から見つける（`finished_timestamp`）．`probe_bypass.py`・`probe_bypass_pullback.py`・`probe_dependence.py`・`probe_dependence_map.py`・`probe_drift.py` の5本は学習済みの重みを要求する（後述）．

`train.py` と `sweep.py` は run の建て方を共有する——`taskalign/training/run.py` の `Settings` と `run_training` である．既定値が報告した実験の値なので，**sweep は変える設定だけを書き，残りが同じであることが目で見える．** `train.py` は冒頭の `DEBUG = True` で，`sweep.py` は `--debug` で，全分割を切り詰めた短い試走になり，数分で配線を確かめられる．`sweep.py` は run を1本ずつ別のプロセスで学習する——run は終わってもホストのメモリを返しきらず，同じプロセスの2本目は1本目の残りを抱えて始まるので，最後の評価の冒頭の山が実行機の RAM（30GB と swap 7GB）を越える．プロセスが終われば，何が抱えていたにせよ全部が返る．山の中身は特定していない．

`train.py` は `SEEDS` の各 seed で baseline → proposal → ablation を順に通す．既定は §8.1 で足した 45・46 で，報告した 42・43・44（`20260829-134859`）は回し直さない．run ごとにモデル・loader・trainer を作り直すので，同じ seed の3群は同じバッチ列を見る（サンプラの epoch カウンタを次の群に持ち越さない）．seed が変わればその並びも初期化も変わるので，**群間の差を run ごとのばらつきと区別できる**．

制約する層は `Settings.layers`，位置は `Settings.positions` で決める．`(16,)` なら第16層の1本，`(8, 16, 24)` なら3本を同時に整合する．位置はプロンプトの末尾から数え，`(-1,)` が報告した実験の末尾のトークン，`(-6, …, -1)` が質問を読み終えた6位置である．層と位置の組ごとに独立に SupCon を計算して**平均**するので，場所を増やしても `align_weight` の意味は変わらない（温度は全部の場所で共有）．

1 run あたり RTX 5090 で約2時間53分（`train_balanced` を1エポック＝29,441 バッチ——全94.3万問のうち，同署名の相方がいない884問はペアを作れないので落ち，残る 942,116 件ぶんを引き終えるまでが1エポック）．**3群 × 3 seed ＝9 run で約26時間．** どちらも `snapshots/20260829-134859_*/log.txt` の実時間．§8.1 の 3群 × 2 seed ＝6 run は，`snapshots/20261105-162351_*/log.txt` の実時間の和で17時間18分，評価まで含めて17時間33分だった．

## 次の一手を走らせる

**研究文書 §8 の六つのうち，§8.1 は走らせ終えた．§8.5 の第1歩（答えが依存する層と位置のマップ）と §8.4 の強さは，`run_plan.sh` のジョブで走らせ終えた（2026-11-25〜26）．** 同じジョブの §8.5 の第2歩（マップが選んだ位置への掛け直し）は，6本のうち1本（proposal・乱数種 42）まで学習したところで，2本目が最後の評価の冒頭でメモリが足りずにカーネルに止められ，ジョブごと落ちた（2026-11-26 11:31）．マップでは，第16層の質問を読み終えた6位置のうち依存が最大なのが整合した末尾の位置だったので，第2歩は規則どおり6位置すべてに掛ける．第1歩と強さの結果は，研究文書の §8.4・§8.5 に書き入れてある（2026-11-26）．

**次のジョブは，§8.5 の第3歩（答えを動かす方向に限った整合），§8.4 の層の項（baseline と，第8層・第16層・第24層に掛けた proposal を乱数種5つで学習し直す），§8.5 の第2歩の順に段を置く（2026-11-27．研究文書 §8 の冒頭）．** 三つは baseline を共有し，第3歩と層の項は第16層の proposal も共有するので，一つのジョブにする．第2歩も，済んだ1本を含めて学習し直す．このジョブはまだ組んでいない——第3歩の固定の射影と，層の項で乱数種と群を広げることと，第3歩の表を読む分析が未実装で，`run_plan.sh` にはまだ段が無い．だから `bash scripts/run_plan.sh position` で第2歩の続きを走らせない．その後の §8.6（モデルを圧縮して冗長さを削ってから整合する）は，削る実装から要る．§8.2（稀な署名を表現の側で）にはまだスクリプトが無く，§8.3（答えの型のラベルで整合し，頑健性も測る）は新しい実装を要する．進捗報告を組む `update_progress.py` は，2026-11-27 のゼミで示した計画（方向を限った整合は，位置を変えた掛け直しが §8.5 第2歩の表の1行目に当たったときにだけ行う）のままである．次のジョブの段を足すときに，次の報告に合わせて組み直す．

```bash
bash scripts/run_plan.sh                   # check → map → strength → position の順に全部
bash scripts/run_plan.sh check             # 機体が整っているかだけを見る（データセット・重み・リモート）
bash scripts/run_plan.sh strength position # 名指した段だけ
```

| 段 | すること | 見込み（RTX 5090） |
| --- | --- | --- |
| `check` | パッケージ・GQA・報告した9本の重み・リモートが，コードの探す場所に在るかを確かめる | 数十秒 |
| `map` | 8問の試走の後，9本（と，重みが在れば第8層・第24層の2本）で相対感度を全層・全位置で測り，第2歩で整合を掛ける位置を選ぶ | 約20分（実測18分） |
| `strength` | 試走の後，$\lambda = 0.03 / 0.3 / 1.0$ を乱数種 42 で学習し，各強さの相対感度・読み取り精度・正解率を並べる | 約9時間半（実測9時間31分） |
| `position` | 試走の後，選んだ位置に proposal と ablation を乱数種3つで学習し，その位置の相対感度と正解率から §8.5 第2歩の表の行を決める | 約18時間 |

**段は，終わるたびに自分で記録する．** `analyses/` の結果と，済んだ run の `snapshots/` と，その時点の結果で組み直した進捗報告（`presentations/20261127_進捗報告.pptx`．`update_progress.py` が書く）を commit し，`git pull --rebase` してから push する．だからノートPCでは，**`git pull` するだけで結果と資料が揃う．** 途中で止まっても，同じコマンドで再開できる．済んだ run は学習し直さない．試走と測定は重みからやり直すが，結果と資料が前と同じなら，その段は何も記録しない．ログは `logs/run-plan-<開始時刻>.log` に落ちる．

**資料は，読みを新しく作らない．** 研究文書が測る前に結果ごとの読みを決めた測定（強さの曲線が単調に下がるか，§8.5 第2歩の表の何行目か）では，`update_progress.py` は，分析が名指した結果に研究文書が決めておいた文を選ぶ．研究文書が記述だけを置く結果（マップ，第16層の6位置での群の差，掛けた層の向き）では，研究文書と同じ判定（区間が 0 を含むか，向き）で文を選び，記述が値に合わなくなれば組むのを止める．ただし研究文書（§8.4・§8.5）に結果を書き入れるのは人の仕事で，それが済むまで，資料は研究文書より先に進んでいる．PowerPoint の無い機で組んだ資料は，開けることを確かめていない（組み方そのものは，同じスクリプトでノートPCの PowerPoint で確かめてある）．

**研究室機では，pull してから，次のプロンプトを Claude Code に貼る．** これは前のジョブ（マップ → 強さ → 第2歩）のプロンプトで，次のジョブを組んだら書き換える．

```text
研究文書 §8 の次の測定（マップ → 整合の強さ → 整合を掛ける位置）を，scripts/run_plan.sh で走らせてください．

読むもの：CLAUDE.md，README.md の「次の一手を走らせる」，scripts/run_plan.sh の冒頭の説明．

すること：
1. ジョブを始める前に一度だけ `git pull --rebase` する．
2. taskalign の入った conda 環境で `python -c "import taskalign, train4all"` が通るかを確かめる．通らなければ，その環境で `pip install -e .` をやり直す．
3. `bash scripts/run_plan.sh check` を走らせる．データセットか重みが見つからないと言われたら，見つからない場所と，実際に在る場所を報告して止まる．
4. `nvidia-smi` で GPU が空いているかを見る．ほかのジョブが使っていたら，始めずに報告する．
5. /ops:exec の分離の形（tmux）で `bash scripts/run_plan.sh map strength position` を始める．
6. map の試走（8問）が通ったことをログで確かめてから，始めた時刻，ログの在処（logs/run-plan-*.log），各段の終わる見込みの時刻を報告する．

守ること：
- コードと設定を変えない．落ちたら，ログの該当箇所と原因の見立てを報告し，直さずに本人に確かめる．同じコマンドで再開できる（済んだ run は学習し直さない）．
- ジョブが自分で行う記録（commit・push）のほかに，commit・push・pull をしない．
- 外部に送らない．予稿（papers/pcsj-imps-2026/）に触れない．

段が終わるたびに報告すること：選ばれた位置と §A.4 の値との差（map），各 run の学習時間と正解率（strength・position），§8.5 第2歩の表の行（position）．
```

**6位置に掛けた run の表現は，git に載らない．** マップが末尾（いま整合している位置）を選ぶと，第2歩は6位置すべてに掛ける．その run の `metrics/representations.pt` は約290MB になり，GitHub が1ファイルに許す 100MB を超えるので，ジョブはこのファイルだけを除いて snapshot を記録する．ファイルは研究室機の `runs/` と作業ツリーに残る．どの位置に掛けたかは `eval.json` の `config.positions` が持つ．

**§8.1（表現の頑健性の向きを決める）は走らせ終えた（2026-11-06）．** 乱数種 45・46 を足した3群の学習が17時間18分（評価まで含めて17時間33分），その重みでの `probe_drift.py` が3時間13分だった（どちらも研究室機の RTX 5090 での実時間）．結果は `analyses/` に，6 run は `snapshots/20261105-162351_*` に在る．回し直すなら，次の順に通す．

```bash
# 1. 学習 — train.py の SEEDS は既定で 45・46．まず DEBUG = True で数分の試走をして，
#    出来た *_debug の run を runs/ と snapshots/ から消してから本走
python scripts/train.py
# 2. 最初に print された timestamp を taskalign/analysis/experiment.py の ADDED_TIMESTAMP に入れる
# 3. 測定 — 既に測った 9 run は analyses/ から読み直し，新しい 6 run だけを測って 5本で比べる
python scripts/probe_drift.py
# 4. 読み直し — 15 run の問ごとの距離を読む（GPU 不要）
python scripts/normalize_drift.py
# 5. ポスター — 研究文書 §8.1・§A.6 に読みを書いてから，手元の機で（要るのは lxml だけで，torch は要らない．
#    taskalign 環境には入っていないので，無ければ pip install lxml か pip install -e .[poster]）
python scripts/update_poster.py
```

`probe_drift.py` が既に測った run を測り直さないのは，同じ条件で測ったと確かめられるときだけである——学習のバッチ（乱数種ごとの timestamp）・checkpoint・加工の一覧・バッチの大きさが同じで，報告用データの全問を測ってあること（`on_file`）．`LIMIT` を入れた試走は，本走が測るはずの run だけを測って表示し，何も書かない．比べる乱数種は `taskalign/analysis/experiment.py` の `ROBUSTNESS_SEEDS`（42–46）で，主結果（§5）を出すほかの分析は `SEEDS`（42–44）のまま読む．区間の係数（`analysis/statistics.py` の `t_critical`）は，乱数種の本数によらず出せる（5本なら 2.776）．

**`probe_drift.py` を走らせる前に**，重みが在るかを確かめる（`python -c "from taskalign.analysis import ADDED_SEEDS, require_checkpoints; require_checkpoints(ADDED_SEEDS)"`．無ければ即座に落ちて場所を告げる．引数を省けば 42–44 を確かめる）．冒頭の `LIMIT` を 64 にすれば数分の試走になる．1 run につき 25 回 testdev を通す——加工 23 通り，対照（同じ質問に別の画像）1回，加工前1回で，各回は生成を含めて学習時の testdev 評価と同じ処理である．加工の定義は `taskalign/analysis/perturbations.py` の冒頭に，Wani らの原文で決まっている部分とこちらで決めた部分を分けて書いてある．

**画像を加工したときの表現の動き（研究文書 §A.6）は測ってある．** `probe_drift.py` を研究室機（RTX 5090）で `LIMIT = None` のまま走らせ，`analyses/probe-drift.json`（run ごとの集計と，群間の対にした差）と `analyses/probe-drift.pt`（問ごとの距離・答えの変化・正誤）を書いた．1 run は 25 pass で約32分，9 run で約4時間53分だった．§8.1 で足した6 run も同じ条件で測り，いまの2ファイルは 15 run ぶんである．`normalize_drift.py` は `.pt` と snapshots だけを読むのでノートPCで回る．

**「利用が伴っていない」の測定（研究文書 §A.4）は3段とも済んでいる**——`probe_bypass.py`・`probe_bypass_pullback.py`・`probe_dependence.py` が `analyses/` に結果を書いているので，走らせ直す必要はない．

**段を手で走らせるなら，ジョブと同じ順に通す．** 層は振ってある——`20260830-223056` の2 run（第8層・第24層，seed 42 の1本ずつ）で，`summarize_sweep.py` はいまこれを読む（研究文書 §A.5）．強さは $\lambda = 0.03 / 0.3 / 1.0$ を seed 42 の1本ずつ回す（既に測ってある $\lambda = 0$ と 0.1 は走らせ直さない）ので約9時間．**強さと層を同時に振らないこと**——一つの曲線に二つの説明が混ざる．`summarize_sweep.py` が読むのは snapshots と GQA の testdev の質問ファイル（と，在れば `analyses/probe-dependence-map.json`）だけなので，重みも GPU も要らない．学習した run の timestamp は，どこにも書き写さない．

```bash
# §8.5 第1歩 — 学習は要らない．--prompts 8 は何も書かない試走
python scripts/probe_dependence_map.py reported --prompts 8
python scripts/probe_dependence_map.py reported        # 9本を測り，第2歩で掛ける位置を選ぶ
python scripts/probe_dependence_map.py layer          # 第8層・第24層の2本（重みが在れば）
# §8.4 の強さ — まず --debug で数分の試走．出来た *_debug の run を runs/ と snapshots/ から消してから本走
python scripts/sweep.py strength --debug
python scripts/sweep.py strength
python scripts/probe_dependence_map.py strength       # 各強さの相対感度
python scripts/summarize_sweep.py
# §8.5 第2歩 — マップが選んだ位置に，proposal と ablation を乱数種3つで
python scripts/sweep.py position --debug
python scripts/sweep.py position
python scripts/probe_dependence_map.py position
python scripts/summarize_relocation.py
# 資料 — analyses/ にある結果だけで組む（どの機でも走る）
python scripts/update_progress.py
```

## 結果を読む

結果は `cfg.runs_root/<timestamp>_<arm>_seed<seed>/` に落ちる．`train.py`・`sweep.py` を1回走らせて回す run は同じ timestamp を共有するので，1回の実験がひとまとまりに並ぶ．

| | |
| --- | --- |
| `config.json` | trainer に渡した引数だけ．taskalign 由来のもの（arm・align_weight・init_scale）は常に，train4all 由来のもの（seed・batch_size…）は既定から変えたときだけ載る |
| `metrics/eval.json` | 群の identity（arm・seed・align_weight・align_pairing・phase）＋testdev の全指標 |
| `metrics/predictions.json` | 公式形式の予測．公式評価器で再採点できる |
| `metrics/representations.pt` | **testdev の表現そのもの**（`(N, L, H)` float32）＋層番号・署名・質問ID．同署名／異署名の cosine 差は 0.02 程度しかなく，半精度ではそれが丸めに沈むので単精度で置く |
| `metrics/epoch_metrics.json`・`metrics/step_metrics.json`・`plots/` | 学習中に記録した指標の推移．JSON と，同じものを描いた PNG |
| `checkpoints/best.pth` | val accuracy が最良のエポックの重み．extras にモデルの checkpoint・層・attention 実装・loader の設定 |
| `log.txt` | 実行の記録．冒頭近くに環境バナーがあり，マシンに続けて結果を決めるライブラリの版が載る（`transformers`・`train4all`・`schedulefree`，そして入っていれば `flash-attn`） |
| `dashboard.html`・`dashboard_data.json` | 学習中のライブ表示と，それが描くデータ．Environment パネルはバナーと同じものを読む |

**同じものの写しが，`checkpoints/` だけを抜いて `snapshots/` の同じ run 名の下にも落ちる．** 重みを落とせば1 run は数十MB——上の表のうち `checkpoints/best.pth` 以外は全部こちらに在るので，git がそのまま追跡でき，push すれば結果は GitHub からも読める．エポックごとと最終評価の直後に，変わったファイルだけを原子的に置き換えて更新するので，走っている最中に掴んでも写しは丸ごと揃っている．

**逆に言えば，重みは clone に付いてこない．** 本走9 run と §8.1 の6 run の `checkpoints/best.pth` は，それを回したマシン（`log.txt` の環境バナーによれば Ubuntu 26.04・RTX 5090）の `runs/` にしかない．`probe_bypass.py`・`probe_bypass_pullback.py`・`probe_dependence.py`・`probe_drift.py` はこれを要求するので，そのマシンで走らせるか，重みを先に持ってくる．持ってくるなら，実行機の `runs/` の写しを USB（`E:\runs`）に置くか，置き場所を `$TASKALIGN_RUNS_ROOT` で名指しする——重みを読む側は run ごとに，`$TASKALIGN_RUNS_ROOT`・ローカルの `runs/`・`E:\runs` の順に探す（`taskalign/config.py` の `run_roots`）．

`snapshots/` には5つのバッチが在る．研究文書の主結果（§5）が報告するのは `20260829-134859` の9 run（3群 × 3 seed）で，分析スクリプトは `summarize_sweep.py` と，§8.1 の乱数種 45・46 も読む `probe_drift.py`・`normalize_drift.py` を除いてこれしか読まない．`20261105-162351` の6 run は §8.1 で足した乱数種 45・46（3群 × 2 seed）で，主結果には入れず，`probe_drift.py`・`normalize_drift.py` だけが読む．`20260830-223056` の2 run は層を第8層・第24層に移した試走（研究文書 §A.5）で，`summarize_sweep.py` だけが読む．`20260727-002344` の9 run は同じ実験を旧版のコードで走らせたもので，走り直す前の報告の記録として残してある．`20260714-033208` の3 run は seed を振る前の単一 seed の実験で，[presentations/](presentations/) の前期公開（2026-07-24）のポスターの裏付けとして残してある——**どちらも別の実行なので，数値が研究文書と一致しなくて当然である．**

**2026-08-30 までの run の名前と `log.txt` の時刻は，実際より先を指している．** 実行機の時計が進んでいたためで，修理したときの記録では約9時間である（確かめられるのは向きだけ——`20260830-223056` の2 run は，名前の時刻より前の 2026-08-30 19:23 に記録されている）．名前は識別子であって測定値ではないので，直していない．時計は 2026-08-30 に NICT へ同期させてあり（実行機の `/etc/chrony/sources.d/nict.sources` と `/etc/chrony/conf.d/authselectmode.conf`．理由はそのコメントにある），`20261105-162351` の名前は実時刻である．

再現に要る情報は3箇所に分かれる——**実験を決める trainer の引数が `config.json`，モデルと loader の設定が `best.pth` の extras，マシンとライブラリの版が `log.txt` の環境バナー**．`config.json` を引数だけに保つので `Trainer(vlm, processor, **config)` がそのまま通り，trainer が受け取らないもの（`vlm` の引数・loader の設定）は extras に回り，**この checkout の外が決めるもの**はバナーが引き受ける．ソースが決めているもの（重みの fp32・bf16 autocast 等）はどこにも無い．この checkout のコードが答えるからで，二重に持たない．

**報告する9 run（`20260829-134859`）は，この checkout のコードで走らせた．** だから上の表と，この3箇所の記述は，そのままこの9 run に当てはまる——`config.json` は `Trainer(vlm, processor, **config)` にそのまま通り，extras にはモデルの checkpoint・層・attention 実装・loader の設定が入り，環境バナーにはライブラリの版が載る．旧版のコードで走った2バッチ（`20260727-002344`・`20260714-033208`）は記録の形がここに書いたものと一部違うが，どれも測定値ではないので，各バッチが当時報告した数値には効かない．

決定指標は `eval.json` の `official_gqa.accuracy`——GQA 公式評価器そのものが `testdev_balanced` に出した数字で，`binary`／`open`／`distribution` と型別内訳が付く．3群の横並びは `summarize_runs.py`（全 testdev の平均・spread・対にした差と区間）と `stratify_by_frequency.py`（署名の頻度で層別したもの）が出す．区間は研究文書 §4.3 の式そのままである（実装は `analysis/statistics.py`）．

`intra_sim`／`inter_sim`／`separation`／`separation_d` は**操作確認**で，決定指標ではない（理由は研究文書）．複数層を制約したときは，学習中は層をまたぐ平均と `separation_d/L16` のような層別が並び，`eval.json` では `representation_stability` が `L16`／`L24` と入れ子になる（単一層ならどちらも平のまま）．`representation_stability` は seed で決まる4,000件の抜き取りを一度に計算したもの——類似度行列が件数の2乗で膨らむので上限を置いてある（`MAX_STABILITY_SAMPLES`）．同じ4つの名前が `eval.json` の `test_metrics` にも並ぶが，そちらはバッチごとに計算してエポックで平均した別の量である．

損失は3つに分かれる．群間比較に使えるのは `lm_loss` だけで，**`loss` ではない**——`loss` は目的関数で，整合する群ではそこに補助項が乗っている．学習後に `train_eval`（train を eval モードで測ったスライス）と `val` を並べて出すので，その2数字の開きが素の過学習量になる．

## 構成

```
taskalign/
  config.py          cfg — デバイスと，2つの根（リポジトリ／作業領域）から引くパス
  types.py           VLM / VLMProcessor のプロトコル．モデル実装との唯一の契約
  pairwise.py        pairwise_cosine / pairwise_equal（損失と指標の共有土台）
  training/          ── 実験を回す
    run.py             Settings / run_training — 1 run の建て方．train.py と sweep.py が共有
    trainer.py         Trainer — train4all の BaseTrainer を実装．3群の定義（Arm）はここ
    losses.py          supervised_contrastive_loss
    metrics.py         Evaluator — run が測る数値はすべてここ．公式 GQA 評価器もここから
    logit_scale.py     LogitScale — 学習可能な温度
  analysis/          ── 済んだ実験を読む（学習は要らない）
    experiment.py      報告する実験はどれで，その成果物をどう読むか（重みの読み口もここ）
    probing.py         線形プローブと，質問文だけを入力にした対照（コードでは surface）
    subspaces.py       部分空間の中と外で読み，同じ幅のランダムな対照と突き合わせる
    jacobian.py        第16層から読み出しへのベクトル–ヤコビ積．§A.4 第2段・第3段が共有する（要 checkpoints・GPU）
    dependence.py      相対感度を全層・全位置で測ってプロンプトの並びに揃え，次に整合を掛ける位置を選ぶ（要 checkpoints・GPU）
    perturbations.py   Wani らの画像の加工（付録 A.6 の族と値どおり．指示文だけ逸脱）と，同じ質問に別の画像を付ける対照
    drift.py           加工の前後で表現がどれだけ動くかを，対照に対する比と Cohen's d で測る（要 checkpoints・GPU）
    statistics.py      平均±標準偏差・対にした差・信頼区間——seed のまとめ方の唯一の定義
  models/            SmolVLM + SmolVLMProcessor（プロンプト構築・ラベルマスク・表現のプール）
  data/
    gqa.py             GQADataset / MPerSignatureSampler / task_signature / build_gqa_dataloader
    _gqa_eval.py       GQA 公式評価器（上流＋局所パッチ．読み方は冒頭のヘッダ）
  utils/             JSON の読み書き・run ディレクトリ名
scripts/
  train.py                  3群を SEEDS の各 seed で本走．回す範囲は冒頭の定数，run の中身は Settings
  sweep.py                  報告した実験が固定した設定を1つだけ振る（整合の強さ・層・位置）
  generate.py               モデルが読み込めて受け答えするかだけを見る最小の確認
  audit_dataset.py          研究文書が語るデータ側の数値を一度に出す（本走に依存しない）
  summarize_runs.py         9 run の記録から研究文書が語る run 側の数値を一度に出す
  summarize_sweep.py        振った設定ごとに読み取り精度と正解率を並べ，整合なしとの対にした差を出す
  probe_signature.py        保存済みの表現から署名が線形に読めるかを測る
  probe_operators.py        プローブが見ていない演算子の組み合わせが読めるかを測る
  stratify_by_frequency.py  済んだ実験を署名の頻度で層別し直す
  probe_bypass.py           整合した構造が出力の読む方向に載っているかを測る（要 checkpoints）
  probe_bypass_pullback.py  その方向を第16層へ引き戻して同じ判定を掛け直す（要 checkpoints・GPU）
  probe_dependence.py       答えが第16層にどれだけ頼っているかを群間で比べる（要 checkpoints・GPU）
  probe_dependence_map.py   それを全層・全位置で測り，次に整合を掛ける位置を選ぶ（要 checkpoints・GPU）
  summarize_relocation.py   選んだ位置に掛けた run の相対感度と正解率から，§8.5 第2歩の表の行を決める
  probe_drift.py            画像を加工したとき表現がどれだけ動くかを，5本の乱数種で群間で比べる（要 checkpoints・GPU）
  normalize_drift.py        その距離を，対照の分布の形と別の比の作り方で読み直す
  update_poster.py          その結果を，§8.1 の読みとともにポスターの3.3節へ書く（torch 不要）
  update_progress.py        analyses/ にある結果から，11/27 の進捗報告を組む（torch 不要）
  run_plan.sh               §8.5 第1歩・§8.4 の強さ・§8.5 第2歩を一つのジョブで走らせ，段ごとに記録する
  check_tidy.py             tmp/ の形と __pycache__ を確かめる（CLAUDE.md の決まりの機械の検査）
```

**`taskalign/` は，実験を回す側と，済んだ実験を読む側に分かれる．** `train.py`・`sweep.py` は `training/` を，分析の13本は `analysis/` を使う——だから import がそのスクリプトの立ち位置を語る（`generate.py` はモデルを読むだけ，`update_poster.py`・`update_progress.py` は資料を書くだけ，`check_tidy.py` は tmp/ を見るだけなので，どれも要らない）．`sweep.py` だけは両方を使う——どの run が済んでいるかと，マップが選んだ位置を，`analysis/` から読む．`analysis/` のうち学習済みの重みと GPU を要するのは `jacobian.py`・`dependence.py`・`drift.py` で，重みだけを読む口は `experiment.py` にある．残りはノートPCで完結する（`normalize_drift.py` は Cohen's d を `training/` の `metrics.cohens_d` から借りる）．

**機構は，この研究の中身に依存しない．** 別の研究に持っていくとき差し替わるのは3つだけ——**ファイルがどこにあるか**（`config.py`），**1 run をどう建てるか**（`training/run.py`），**どの実験を報告するか**（`analysis/experiment.py` の `TIMESTAMP` と `ADDED_TIMESTAMP`）．残りはすべてパスを引数で受け取り，実験を名指ししない．だから `taskalign/` で `cfg` を import するのは後の2つと，済んだ実験の重みを読み直す `analysis/jacobian.py` だけで，機構は一つも import しない——`data`・`models`・`types`・`pairwise` と，`training/` の残り（`trainer.py`・`losses.py`・`metrics.py`・`logit_scale.py`）のどこにも `cfg` は無い．`taskalign/__init__.py` が何も再輸出しないのも同じ理由で，`import taskalign` がマシン固有のパスを束縛しないためである．

`cfg` 自身も2種類を分けて持つ——**マシン固有**（データセット・`runs/`・デバイス）と，**チェックアウト相対**（`snapshots/`・`analyses/`）．**済んだ実験を読む側が結果を書くのは後者だけ**である．マシン固有の側から読むのは，GQA の質問ファイルと，重みを要するスクリプトが使う `runs/`・GQA の画像・デバイスに限る．

**`scripts/` は走らせるものだけを置く．** 複数のスクリプトが分け合うもの——どの実験を読むか，成果物の在処，プローブの当てはめ——は `taskalign/` 側にある．スクリプトが自分で決めてよいのは，自分だけの設定と，出力の並べ方だけである．

**置くのは，学習・検証と，環境が整っているかの最小の確認に要るコードだけ．** ほかのスクリプトの出力に含まれる情報を再計算するだけのものは持たない．ルート直下のディレクトリ名は小文字の複数形の名詞にする．

**run が測る数値は `metrics.py` だけを見ればいい．** デコードも採点も表現の比較も公式評価器もそこにあり，`Evaluator.measure()` が1バッチの，`Evaluator.report()` がエポック全体の測定値を返す．Trainer が足すのは自分が組んだ目的関数の内訳（`lm_loss`・`align_loss`・`temperature`）だけで，モデルを測ることはしない．**`analysis/` が測るのはこれの例外ではない**——問うのは run が残したもの（表現・予測・重み）であって，走っているモデルではない．

**モデルは `models/` を足すだけで差し替わる．** `types.py` の `VLM` / `VLMProcessor` プロトコルが唯一の契約で，`VLMOutput["representation"]`——整合する層と位置の組ごとにプールした `(B, L, H)`——を返せば，loader も損失も指標も Trainer もそのまま動く．

**整合を表現の一部だけに掛けたくなったときの差し込み口は `Trainer._project`**（現状 identity——表現をそのまま返す）．入れるなら**固定の**射影にすること——学習可能にすると射影の側を動かすだけで損失を満たせてしまい，仮説が反証不能になる．

**レバーは二つある．** 層を選ぶ側（`Settings.layers`）は，`sweep.py layer` が第8層・第24層を seed 42 で走らせてある．位置を選ぶ側（`Settings.positions`）は，プロンプトの末尾から数えた位置を渡す．`sweep.py position` は，マップの選んだ位置（6位置すべて）に掛けて，proposal と ablation を乱数種3つずつ学習する．6本のうち1本まで済んでいる．複数の層に同時に掛ける学習は，`Settings.layers` に複数の層を渡せば走る（場所ごとの損失の平均）．こちらは実装してあり，未実行である．分析の側は一つの層に掛けた run しか読まない（`analysis/experiment.py` の `constrained_layer` が断る）ので，読むには手を入れる．複数の位置に掛けた run では，`representations` が最初の位置の表現を返す．

**研究文書 §8.3，§8.5 の第3歩，§8.6 が計画する学習は，未実装である．** 第3歩の $J(U_{32})$ への固定の射影（`Trainer._project` に入れ，同じジョブで学習した同じ乱数種の baseline から求めて渡す），§8.3 の答えの型をラベルにする学習，§8.6 の枝刈り（言語側の注意ヘッドと MLP の中間のニューロンを削る．残す基準は未選定）である．

## 研究文書の記述は，コードのどこか

節番号は研究文書のもの．

| 研究文書 | 実装 |
| --- | --- |
| $\mathcal{L}_{\mathrm{align}}$（§3.2） | `losses.supervised_contrastive_loss`——対角を `-inf` にした `pairwise_cosine(features) * logit_scale` に `logsumexp` を取り，正例の平均を引く |
| 温度 $\tau$（§3.2） | `LogitScale` が $1/\tau$ を log スケールで保持し 100 で clamp．初期値は `logit_scale.INIT_SCALE = 1/0.07` |
| $\mathcal{L}_{\mathrm{LM}}$ のマスク（§3.2） | `SmolVLMProcessor._mask_labels`——プロンプト位置と `attention_mask == 0` の位置を `IGNORE_INDEX` に落とす |
| 掛ける位置（§3.3） | `SmolVLM._pool`——`hidden_states[row, prompt_lens[row] + position]`．`position` はプロンプトの末尾から数え，報告した実験は `-1`（`Settings.positions`） |
| プローブが読む表現（§A.1） | 損失が使うのと**同一のテンソル**．`Trainer` が `set_cache("representation", …)` した値を `Evaluator.measure` がそのまま貯め，`representations.pt` に `(N, 1, H)` で落ちる．**評価ローダも `with_labels=True`**（`training/run.py` の共有 kwargs）で右詰め——左詰めだと `prompt_lens` が作られず，`_pool` の位置が意味を失う．生成は `decode_answers` が質問だけを左詰めに詰め直した別テンソルで行うので，表現には触れない |
| 分離度 $\delta$ の $\sigma$（§4.4） | `metrics.cohens_d`——`torch.var` は不偏なので，文書の式どおり $n-1$ で割った分散から作る |
| $\mathcal{P}$ と $\mathcal{N}$（§4.4） | `metrics.representation_stability`——対角を除いた類似度行列**全体**から取るので順序つき対になる．平均と分散は変わらない |
| プローブ（§A.1） | `analysis/probing.py`——sklearn は入れていないので torch の線形層＋AdamW で解く．当てはめ方と打ち切りの根拠はモジュール冒頭．研究文書の**言い回しプローブ**が `surface_matrices`（コードでは一貫して `surface`）で，`words`／`words_and_pairs` の高いほうを採る |
| **形式の残差化**（§A.1） | `probing.format_matrix` が形式の3特徴を標準化して並べ，`residualize` が ridge の当てはめを引く．**署名ラベルは回帰に入らない**ので，プローブの学習側・評価側を分ける前の全行で当てはめてよい．残差を再正規化しない理由は関数の docstring にある |
| 表現の幾何・損失・正解率の要約，崩壊値 $\log(B-1)$，1エポックのバッチ数と実行時間（§4.1・§4.4・§5.2） | `summarize_runs.py`——9 run の `eval.json`・`step_metrics.json`・`log.txt` を読み直すだけ．新しく測るものは無く，文書がしている算術をここでする |
| $\bar{d}$・信頼区間（§4.3） | `analysis/statistics.py`——係数 $t_{n-1, 0.975}$ は表を引かずに求める．自由度2（乱数種3つ）は文書の 4.303 を与える閉じた式で，それ以外は不完全ベータ関数を数値的に逆に解く．区間が 0 を跨ぐか否かも保存する |
| 頻度層別と層ごとの区間（§A.3） | `stratify_by_frequency.py` |
| 整合の強さ・層を振った点（§8.4・§A.5） | `sweep.py` が点を作り，`summarize_sweep.py` が並べる——読み取り精度は §5.1 と同じ当てはめで測り直すので，既存の2点が §5.1・§5.2 の数値を再現することが，両者が同じものを測っている証拠になる．各点の相対感度はマップ（次の行）から読み，強さの曲線は乱数種 42 の値だけで作って，幅 $m = 32$ の相対感度がどの段でも下がるかを残す |
| 答えが依存する層と位置のマップ，掛ける位置の選択（§8.5 第1歩） | `probe_dependence_map.py`——§A.4 第3段と同じ方向・プロンプト・逆伝播で，全層の出力について勾配の長さを取る（`analysis/jacobian.py` の `vjp_everywhere`）．位置は，質問より前の70位置と質問文を先頭から，質問を読み終えた6位置を末尾から数えて揃える（`analysis/dependence.py`．70 と 5 は `SmolVLMProcessor.prompt_layout` がトークナイザで測る）．選択の規則は `choose_positions`，読む幅は `PRIMARY_DIM`．第16層の末尾の値を `probe-dependence.json` と突き合わせた差も残す |
| 選んだ位置への掛け直しと，表の行（§8.5 第2歩） | `sweep.py position` が proposal と ablation を学習し，`probe_dependence_map.py position` がその位置の相対感度を測り，`summarize_relocation.py` が乱数種を対にした四つの差（相対感度と正解率の，proposal − baseline と proposal − ablation）から行を決める．掛けた位置の分離度も `eval.json` から並べる |
| 「利用が伴っていない」の測定・第1段（§A.4） | `probe_bypass.py`——checkpoint の出力層から答えの**先頭トークン**の行を抜き，中心化した主成分 $m$ 本を $U$ とする．表現を $U$ の中と外へ射影して同じプローブを掛け，**中と外の両方**を同次元のランダム直交基底と比べる．部分空間の道具は `analysis/subspaces.py` にあり，第2段と共有する |
| 「利用が伴っていない」の測定・第2段（§A.4） | `probe_bypass_pullback.py`——部分空間 $U$ の各方向をベクトル–ヤコビ積で第16層へ引き戻し（`SmolVLM.readout_from_site` が site と読み出しを微分可能なまま返す），直交化して $U$ の代わりに使う．引き戻す前後の cos も測る——1 に近ければ第1段の近似は正しかったことになる（実際は 0.2 台だった） |
| 「利用が伴っていない」の測定・第3段（§A.4） | `probe_dependence.py`——同じベクトル–ヤコビ積の**長さ**を，第16層の表現と読み出しの大きさで割って無次元にし（相対感度），3群で比べる．第2段が正規直交化のときに捨てている量である．プロンプト本数の収束も同じ逆伝播から出す |
| 表現の頑健性の測定（§A.6） | `probe_drift.py`——Wani らの加工（`analysis/perturbations.py`．付録 A.6 の族と値をそのまま使う．書かれていない部分はこちらで決め，指示文の選び方だけは逸脱して，どちらも理由とともに冒頭に明記）を testdev の画像に施し，加工前後の第16層・第32層の表現のコサイン距離を，同じ質問に別の画像を付けた対照の平均に対する比と Cohen's d で出す（`analysis/drift.py`）．答えが変わった対の割合（IFR）・変わった問の割合（IV）・答えが変わらなかった対だけの比も同じ pass から出す．Wani らの形の対照（別の問と別の画像）は加工前の表現だけから出す．d の標準偏差は `metrics.cohens_d`（Wani らはプールの仕方を書いていない） |
| 対照の分布の形，d の分解，別の比の作り方（§A.6） | `normalize_drift.py`——`probe-drift.pt` の問ごとの距離を読む（重みは要らない）．対照の変動係数・歪度・分位点の比，d の合わせた分散に対照が占める割合，標本数で重みを付けない d と Glass の Δ，分布の形を仮定しない重なり，中央値 ÷ 中央値・問ごとに割った比の幾何平均・対照の上位1%を除いた比・同署名の距離に対する比（同署名の類似度は各 run の `eval.json` から読む）．群間は他と同じく `analysis/statistics.py` を通す |
| 分割の排他性・署名の分布・対を組めない問・含意対との一致・答えの語彙・演算子が自分の名を質問文に置く率（§3.1・§4.1・§7・§A.4） | `audit_dataset.py`——学習も GPU も要らず，質問ファイルだけから全部を一度に出す．含意対は**対の数**（`unordered`）で数える．GQA は同じ関係を両側から並べるので，訪問回数（`ordered`）で数えると往復を二度数えて率が数ポイント動く．研究文書が引くのは前者．**食い違いの中身**（yes/no 型と自由回答型を組にした割合，最も多い署名の組）も同じ関数が出す——率だけでは「署名が誤っている」としか読めないので |

**研究文書の数値には，例外なくそれを出すスクリプトがある．** 手で数えて本文にだけ書いた数値は，書いた時点から誰にも検算できない．出どころは主張の種類で決まっていて，新しく何かを主張するならまずその場所に関数を足す．

| 主張の種類 | 出す場所 |
| --- | --- |
| データについて（分割・署名・演算子・語彙） | `audit_dataset.py` |
| 9 run の記録について（幾何・損失・正解率・実行時間） | `summarize_runs.py` |
| 保存済みの表現・距離に新しく問うこと | `probe_signature.py`・`probe_operators.py`・`probe_bypass.py`・`stratify_by_frequency.py`・`normalize_drift.py` |
| 重みを走らせ直して新しく問うこと（引き戻し・相対感度・そのマップ・頑健性） | `probe_bypass_pullback.py`・`probe_dependence.py`・`probe_dependence_map.py`・`probe_drift.py` |
| 報告した9 run の外の run について（強さ・層・位置を変えたもの） | `summarize_sweep.py`・`summarize_relocation.py` |
| seed のまとめ方（平均・spread・区間） | `analysis/statistics.py`——上のどれもがここを通る |

**要約統計を各スクリプトで書き直さない．** 平均・標準偏差・対にした差・信頼区間は `analysis/statistics.py` にしかない——二つのスクリプトが別々に区間を作れば，同じ文書の中で同じ括弧が二つの意味を持つ．
