# Engineering & Validation Roadmap — torch-wgi

Version: 0.1-proposed | 2026-10-06 | Status: PLANNED / NOT RUN

研究管理： [統括 #5](https://github.com/ohayotaro/torch-wgi/issues/5) → [Phase 1 #1](https://github.com/ohayotaro/torch-wgi/issues/1) → [Phase 2 #2](https://github.com/ohayotaro/torch-wgi/issues/2) → [Phase 3 #3](https://github.com/ohayotaro/torch-wgi/issues/3) → [Phase 4 #4](https://github.com/ohayotaro/torch-wgi/issues/4)。

この文書は実装・実証の計画であり、実験結果ではない。数値閾値は本研究で提案するengineering acceptance criteriaであり、規格値、文献の達成値、採択保証ではない。校正用pilotで妥当性を確認し、最終testを見る前に固定する。改定は理由、影響、再検証範囲、新しいholdoutとともに記録する。

## 0. 研究の不変条件

### 0.1 確率モデルとユーザー要求

本研究の対象は、同一崩壊位置でPETとComptonの観測尤度を結合する画像空間モデルである。Cone-conditioned Tube Projectorはその積分の座標変換・周辺化であり、LORとconeの交点座標を代数的に解く実装ではない。

画像係数 $f_j$ は撮像期間中の期待崩壊数、$\varphi_j$ は積分が1の画像基底とする。条件付き独立性を適用できる部分では、

$$A^J_{ij}(\theta)=\beta_3\int \varphi_j(x)P_\theta(y_i^P\mid x)C_\theta(y_i^C\mid x)\,dx,$$
$$\lambda_i=\sum_j A^J_{ij}f_j+b_i,\qquad s_j=\int A_j(y;\theta)\,d\nu(y).$$

有限ボクセルでは同じサブボクセル点で $P C$ を計算してから積分する。一般に $\mathbb E[PC]\ne\mathbb E[P]\mathbb E[C]$。また $(Pf)(Cf)$ は別位置由来のcross termsを含むため、このモデルに置き換えない。

核崩壊点xと消滅点zは陽電子飛程で結ぶ：$P(y^P\mid x)=\int r_\beta(z\mid x)p(y^P\mid z)dz$。共通時刻ゲート、読出し相関、未知散乱順序、未測定DOIは必要な潜在状態に戻して共同応答で積分する。診断用MC truthを観測データへ漏洩させない。

絶対応答を画像空間で正規化しない。lost/非受理確率、分岐比、検出器の遮蔽、dead channels、幾何依存Jacobianを残す。感度sは全可能観測の積分であり、観測済みList-mode行の和ではない。List-mode likelihoodの生成モデルを出発点とする [R5]。

### 0.2 設計の公平性

主比較は線源量・撮像時間・材料量/チャネル数・開口・製造制約を固定する。等検出カウント比較は分解能の副次解析として別に示す。PET-only、Compton-only、tripleは排他的event classとし、同じtripleを独立PETとして二重計上しない。

fに対する作用素の線形性を保持する。イベント組合せのランダム成分やdead timeが活動量に依存する場合は、背景/受理モデルの非線形部分として別途微分し、線形Aと混同しない。

### 0.3 3種類の誤差を分離する

1. 実装誤差：AD/有限差分、随伴、正規化。
2. 数値・近似誤差：求積、cone局所線形化、SDF緩和、SRF近似、sparsification。
3. 物理モデル誤差：散乱多重度、Doppler、減弱、読出し、背景、実機との差。

同じ近似をADと有限差分で微分して一致しても、2と3は検証されない。逆に、高統計MCとの小さい有意差だけで失敗とせず、事前指定の実用的等価marginとCIで判定する。

## 1. Phase 1 — Toy演算子・微分・随伴

### 1.1 実装スコープ

2Dは符号/ビニング診断用とし、主PoCは単純3Dにする。32^3〜64^3、平滑Gaussian/球病変、少数の平面検出器または小型リングから開始。無減弱、既知位置、単一散乱の有限幅応答を先に完成させ、均一減弱と有限検出体積を追加する。

次の3実装を独立に保持する。

- `ReferenceJointProjector`：CPU float64、高次数サブボクセル積分。速度より独立な検証を優先。
- `TubeProjector`：$x=m+tu+B\delta$ 上で有限幅PET×Comptonを直接求積。
- `ConeConditionedTubeProjector`：$g(x)\simeq g_0+b^T\delta$ とGaussian積により横方向を条件付き周辺化。

条件付きモデルでは $S=\sigma_C^2+b^T\Sigma_Lb$、$\mu_\delta=-\Sigma_Lb g_0/S$、$V_\delta=\Sigma_L-\Sigma_Lbb^T\Sigma_L/S$ を使い、$\mathcal N(g_0;0,S)\,\mathbb E_{\delta\sim\mathcal N(\mu_\delta,V_\delta)}[fW]$ をt方向に積分する。微小病変では期待値をLOR中心値だけで代用しない。局所線形化の誤差が大きいときは直接Tube積分に戻す。

`geometry`は制約付きテンソルから検出器位置/寸法/SDFを生成する。材料境界は $\rho_\tau=\mathrm{sigmoid}(-d/\tau)$、検出体積の積分は参照領域への引き戻しとJacobianで表す。光子輸送は同じ材料場の線積分から計算する。

読み出しSoft-binningは

$$S_k=\Phi((b_k-u_\theta)/\sigma_{eff})-\Phi((a_k-u_\theta)/\sigma_{eff}),\quad \sigma_{eff}^2=\sigma_{phys}^2+\tau_{read}^2.$$

物理SRFが既に含まれる段階に再度$\sigma_{phys}$を加えない。$S_{lost}=1-\sum_kS_k$を保持する。実測イベントを分数カウントに分割した後、独立Poissonビンとして扱わない。SDFの空間τと読出しτは別パラメータにする。

### 1.2 計算グラフ・API

必須インターフェースは `project(f, observations, geometry)`, `adjoint(weights, observations, geometry)`, `sensitivity(geometry, observation_rule)`, `jvp_geometry`, `vjp_geometry`。初期の随伴はfについてのVJPで構成する。θによる積分点、重み、有限体積、感度の微分も保持する。疎支持/求積配置をfに依存させる場合、線形作用素として使える条件を別途定義する。

代表的な幾何勾配テストの実装契約：

```python
# Pseudocode: operator and the fixtures are Phase-1 deliverables.
# eta, f, y, quadrature are deterministic CPU float64 tensors.
def objective(eta):
    theta = constrained_geometry(eta)
    return (operator.project(f, obs, theta) * y).sum()

value = objective(eta)
grad, = torch.autograd.grad(value, eta)
g_ad = grad @ direction
for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    g_fd = (objective(eta + h*direction)
            - objective(eta - h*direction)) / (2*h)
    record(h=h, autodiff=g_ad, finite_difference=g_fd)
```

`torch.autograd.gradcheck`と方向微分試験は相補的に使う [R2]。複数のhで安定域を検査し、単一のhでたまたま一致した結果を採用しない。

### 1.3 メトリクスとG1

パラメータは $\theta_k=\theta_{0k}+s_k\eta_k$ で無次元化し、目的も参照値でO(1)にする。

$$g_{FD}(h)=\frac{L(\eta+hv)-L(\eta-hv)}{2h},\quad g_{AD}=v^T\nabla_\eta L.$$

非零方向の相対誤差は $|g_{AD}-g_{FD}|/\max(|g_{AD}|,|g_{FD}|)$。近零方向は絶対基準で判定する。50幾何×20方向を基本とし、ランダム試験に加えて接線近傍、検出器端、gap、薄い結晶、厚いcone、ゼロ近傍の感度を含む決定的fixtureを作る。

| G1項目 | 提案PASS条件 |
|---|---|
| AD–FD | 相対誤差p95≤1e-4、最大≤1e-3。近零は絶対誤差≤1e-8。隣接2段以上のhで確認 |
| gradcheck | 全ての正則な基準fixtureで成功 |
| 随伴 | 正規化内積残差≤1e-10 (CPU f64)、≤1e-5 (採用GPU f32) |
| 縮約vs独立3D参照 | カウント加重L1相対誤差≤0.5%、安定非零方向の勾配誤差≤2% |
| 求積次数/解像度倍増 | 順投影変化≤0.5%、勾配変化≤2% |
| toy確率保存 | 既知の受理+lost正規化で残差≤1e-6、負確率/NaN/Infゼロ |
| τ収束 | 最終2段の目的変化≤1%、非零勾配cosine≥0.99、hard参照との感度/目的差≤1% |

随伴試験は

$$e_{adj}=\frac{|\langle Af,y\rangle_Y-\langle f,A^*y\rangle_X|}{\|Af\|_Y\|y\|_Y+\|f\|_X\|A^*y\|_X+\epsilon}.$$

質量行列で内積を定義する場合は $A^*=M_X^{-1}A^TM_Y$。期待崩壊数係数とEuclidean観測ベクトルの契約なら、体積因子をAに含めて $A^*=A^T$ とする。VJPで自動構成したA*の合格は離散的整合性の証拠であり、物理の正しさの証拠ではない。

### 1.4 τアニーリングとデバッグ

初期案として人工τ/特徴ピッチを0.25, 0.125, 0.0625, 0.03125, 0.015625とする。各段で求積収束を確認してから縮める。局所積分間隔≤τ/4を初期目安とするが、それだけで十分とは仮定しない。物理的な位置/エネルギー/Doppler幅は縮小しない。

小さなτで勾配がゼロなら、境界を積分点が捕捉しているか、候補集合が古いか、sigmoidが飽和しているかを確認する。発散なら未解像境界、共分散、真に非滑らかなhard設計点を分ける。τを戻し局所求積を増やす。緩和極限を安定に扱えない場合は参照検出体積積分または明示的境界寄与の方法へ戻す。

CDF差の桁落ちは安定な裾計算、$1-e^{-a}$は`-expm1(-a)`、小確率の積はlog-domainで扱う。`acos`の端点微分、測定cos値の無条件clip、近接頂点への任意epsilonを主経路に置かない。cone頂点/ゼロ長の不可能幾何は物理制約で除外する。Cholesky失敗はFP64で物理分散・対称性を確認し、jitterの影響を記録する。

Done：全G1条件と失敗領域/フォールバック、テスト、再現manifestをPRで提出し、独立再実行後にG1を閉じる。

## 2. Phase 2 — Geant4/GATEによる物理クロスバリデーション

### 2.1 ベンチマークの階層

まず共通設定から単一スラブ、検出器対、簡略二重リングを生成する。既存WGIの例としてscattererリング20 cm、PETリング66 cmの直径が報告されている [R1]。しかし径だけでは実機を再現できない。結晶材質/寸法/軸長、DOI、gap、dead material、閾値、energy window、coincidence windowの一次資料と設定が揃うまでは「文献を参考にした合成幾何」とする。

物理追加は次の順で行う。

- V0：理想的な同一崩壊3光子、真空、自由電子単一散乱、完全読出し。
- V1：有限結晶、光電吸収/全減弱、単一Compton後の吸収、有限energy/position resolution。
- V2：材料依存Doppler、2回Compton後の吸収、未知順序、escape/atomic relaxation。
- V3：体内減弱/散乱、陽電子飛程・非共線性、実核種の分岐、観測可能なevent selection。
- V4：研究対象に必要なrandoms、dead time、pileup等。未対応なら活動量/計数率の適用範囲を制限する。

2回散乱はsingle-scatter coneを単に太くするモデルではない。経路仮説の和と検出体積積分として実装するか、観測可能なreject条件で抑え、その残差を定量化する。MC truthを使うoracle selectionは診断のみで、実装性能として報告しない。

Geant4はComptonモデルによって束縛電子/Dopplerの扱いが異なる [R3]。physics-listの名前だけでなく、実際に適用されたprocess/modelとenergy範囲を出力・固定する。GATEからのphysics設定は公式APIに従う [R4]。production cutは単なる幾何ステップ長ではないため、cut半減試験と必要な追跡設定の収束を別々に確認する。

### 2.2 共通データ契約

観測テーブル：event/run識別子、event class、検出channel、測定energy/time、実際に取得可能なDOI、readoutの品質flags。

診断テーブル：primary/parent/track識別子、放出点、真energy/位置、全相互作用process、scatter multiplicity、true order。前者に後者の情報を混入させない。

単位はmm/keV/ns、放射能は設定単位から期待崩壊数へ明示変換する。分岐確率の乗算箇所、1崩壊あたりの受理件数、event buildingの規則を固定する。高統計MCでaccepted数だけを合わせて再正規化してはならない。

点線源は中心、径方向0.5/0.8 FOV、軸方向中心/端など事前指定の主要位置へ置き、減弱あり/なしで評価する。energy・散乱角・位置の主要層を事前に固定する。校正用と未使用geometryのvalidationを分離し、testの残差を見ながらSRFをfitしない。

### 2.3 誤差、統計、合否

rateとshapeを分ける。絶対感度は $\epsilon=N_{accepted}/N_{decay}$。形状は共通readout bin上の $p_b=\lambda_b/\sum\lambda_b$ とし、$TV=\tfrac12\sum_b|p_b^{ana}-p_b^{MC}|$ で比較する。形状が合っても総感度が違えばFAIL。

| G2主要指標 | 提案等価margin |
|---|---|
| 単純スラブ確率 | 相対1% |
| 絶対感度・受理総計数 | 各主要位置/classで±5% |
| 正規化count shape | TV≤0.05 |
| 主要readout/energy/angle層のcount | ±10% |
| energy peak中心 | 基準energyの±1% |
| energy FWHM・ARM FWHM | ±5% |
| ARM重心ずれ | 基準ARM FWHMの0.1倍以内 |
| 空間FWHM | 理想SRF±5%、全応答/同一再構成±10% |
| FWTM・90/95%幅 | ±15%、事前tail確率の差≤0.02 |
| cut/求積収束 | 主要指標変化≤1% |

FWHMは中心・径/接線/軸方向を分け、同じfit/補間/再構成法で測定する。非Gaussian形状ではFWHMだけでなくtailとquantile widthを主要項目にする。sourceの有限径とvoxel幅を両系で合わせ、1 voxel未満の差を安易に物理改善としない。

提案統計設計：全体感度のMC相対標準誤差≤0.5%、主要層の95%相対半幅≤2%。単純な非加重rare-eventなら約4万accepted eventsで前者の目安になるが、必要な発生崩壊数は感度次第。weight、同一primaryの相関、split historyを考慮し、history単位のbatch/bootstrapでCIを計算する。

符号付きrelative differenceの95%CI全体が $[-margin,+margin]$ 内ならPASS。正の距離指標では上側CIが閾値以下を要求。CIが広すぎるだけならINCONCLUSIVEとしてMCを増やす。統計誤差を閾値に足して許容誤差を拡大しない。主要多層比較では同時CIまたは事前指定の階層的検定を使う。

### 2.4 応答値だけでなく、設計方向微分をMCで検証

5以上の幾何近傍と10以上の非零方向について、

$$g_{MC}(v;\delta)=\frac{Q_{MC}(\theta+\delta v)-Q_{MC}(\theta-\delta v)}{2\delta}$$

を測る。共通乱数は共通のprimary条件を使い、追跡の分岐で相関が失われ得ることを踏まえて分散を実測する。$\delta/2$等で有限差分biasを確認し、小さくしすぎてMC noiseに埋もれた方向を採用しない。

95%CIがゼロを除外する十分な統計の方向で、符号一致≥90%、大きさ誤差median≤15%、p90≤30%を初期ゲートとする。全設計変数を識別可能な方向で覆う。underpowered方向を合格に含めず、重要な失敗方向があれば適用領域を狭めるかモデルを修正する。

G2Aは定義した単純物理、G2Bは最適化に使う受理全応答のゲート。未モデル化受理成分は上側95%CI≤1%に加え、その成分の追加/除外によるタスク変化≤2%を要求する。小さいcount fractionでもタスク重要度が大きければ無視しない。

### 2.5 性能比較とフォールバック

速度は同じ期待ヒストグラム/感度/タスク量を同じ精度で得る仕事について比較する。MCの輸送events/sと解析モデルの尤度events/sの直接比は採用しない。wall-clock、CPU core-hours、GPU-hours、RAM/VRAM、cold/warm、I/O、校正MC、前処理を分けて示す。実用目標10x、stretch100xは未達成の目標であり、G2の物理合格とは別判定にする。

差が大きいときは単位→分岐/同時計数→材料/減弱→scatter multiplicity→readout→SRF→再構成の順に切り分ける。physics-list間の差はMCのmodel discrepancyとして記録する。高次散乱が支配的なら、少数経路の決定論的積分と物理的に制約された補正成分へ戻す。G2Aのみならsingle-scatter PoCとして範囲を限定する。

Done：未使用geometryでG2Bの値と方向微分がPASS、MC再生成設定/seed/data hashと残差レポートが公開可能、未対応物理と適用範囲が固定される。

## 3. Phase 3 — タスク駆動ハードウェア最適化

### 3.1 目的関数と評価対象

$f=f_0+\alpha\psi$、$D_\alpha=A_\theta\psi$ とし、固定観測測度で

$$I_{ab}(\theta)=\int \frac{D_a(y)D_b(y)}{\lambda_0(y)}d\nu(y),$$
$$I_{eff}=I_{\alpha\alpha}-I_{\alpha\nu}(I_{\nu\nu}+\Lambda_{prior})^{-1}I_{\nu\alpha}.$$

priorは事前知識として固定し、悪条件を隠す自由な性能向上パラメータにしない。数値計算はinverseではなくsolve。目的は例えば $-\mathbb E[\log(I_{eff}/I_{ref})]+\gamma C(\theta)$ とし、病変/背景/製造公差の分布を事前定義する。

Poisson list-modeの情報量は観測モデルと結び付く [R5]。実測固定リストの再構成損失だけで、異なる装置の将来データ分布を比較しない。受理数で条件付けた正規化尤度だけでは感度情報を失う。

観測求積は固定readout座標で定義する。重要度密度qを用いるなら重み1/qと支持被覆を保持する。θ依存のサンプル変換ではJacobian/測度の微分を正しく扱う。未知DOI/順序を周辺化してからFisherを計算する。MCの粗いbinで得たFisherと解析の連続Fisherをそのまま一致判定しない。同じbinへの積分結果を比較した後にbin収束を調べる。

### 3.2 1変数実験

厚みまたはリング間隔を1つ選び、可行範囲を固定する。61点の高精度走査とピーク/境界周辺の細分化をreferenceとする。8初期値からAdam等で最適化し、

$$regret=\frac{I_{ref}^{max}-I(\hat\theta)}{I_{ref}^{max}}\le 0.01$$

を要求する。reference積分誤差はこのmarginより十分小さくする。平坦な最大領域では座標一致ではなくnear-optimal集合への到達を評価する。

感度と分解能のトレードオフは解釈に使うが、内点最適の存在を仮定しない。単調な性能曲線や制約境界での最適も有効。9〜13の事前指定幾何と最終候補をMCで確認し、解析モデル自身の直感で成功判定しない。

### 3.3 多変数と局所解

16以上のSobol/LHS初期値を使い、τ継続、trust region、局所Adam/L-BFGSを比較する。最良値だけでなく全開始点の分布と失敗率を公開する。正規化ηを使い、パラメータの単位差でoptimizerを不利にしない。

完全リングでN固定、pitchを弧長で定義する場合はNp=2πRを守る。結晶幅/gap/半径を独立指定するなら幾何閉包条件を実装する。N等の整数変更は外側の候補探索を主とし、occupancy gateによる連続トポロジー緩和は別実験にする。緩和で得た仮想物質は、二値化→連続再最適化→hard geometry MCで再検証する。

MC anchorは初期、探索途中、最終、公差摂動に置く。解析値または方向微分の検証が崩れたらtrust regionを縮小し、必要なモデル校正を行う。校正・探索・モデル選択・最終testを分離し、MC anchorを最終testの代わりにしない。

### 3.4 MLEMと病変評価

初期/最終のhard geometryに対して独立MC list-modeを生成する。再構成は

$$f_j^{n+1}=\frac{f_j^n}{s_j(\theta)}\sum_i\frac{A_{ij}(\theta)}{\sum_kA_{ik}(\theta)f_k^n+b_i(\theta)}.$$

背景の期待総数も含むPoisson目的をログに保存する。感度は全観測積分、A*は採用離散化の随伴。画像格子、補正、停止/postfilter規則はvalidationで固定する。単なる同一iteration比較に加えてCRC-noise曲線を示す。

$$CRC=\frac{\bar f_L/\bar f_B-1}{f_L^{true}/f_B^{true}-1}.$$

CNRは定義を固定する。例えば病変ROIと対応背景ROIの平均差のensemble平均を、同じ統計量の背景仮説での標準偏差で割る。背景ROI内のvoxel標準偏差だけをnoiseとすると空間的不均一を混同するため、独立noise realizationsを使う。

主要検出指標は、固定されたROI検出スコアまたは独立train/testしたCHOに対するd'またはROC AUCとする。CRC/CNRは重要な副次指標だが検出能の唯一の証拠にはしない。病変サイズ、コントラスト、中心/周辺/軸端、背景不均一、減弱誤差、公差をholdoutへ含める。

まず病変あり/なし各100程度をpilotとして分散を推定し、主要効果とCI幅で本試験の必要標本数を事前設定する。この100は十分性を保証する数ではない。phantom/primary/noise共有による相関を反映し、testを見ながら有意になるまで任意停止しない。

### 3.5 G3と不成功時の扱い

同一readoutの解析FisherとMC参照の差が±10%以内、1D regret≤1%、製造制約を全て満たすことを数値ゲートとする。

設計改善の実用目標は主要d'/Fisher等の相対改善点推定≥10%、改善の95%CI下限>0。AUCを主要指標にする場合は相対10%ではなく、pilot後に固定する絶対差（例0.03）とCIを用いる。主要位置/背景層では相対5%の非劣性marginを提案するが、AUCのmarginは絶対差で別設定する。

予測改善が独立MCで消えるなら、最適化成功ではない。SRF裾、未知相互作用、感度、緩和gap、観測支持漏れ、背景/nuisanceを順に診断する。改善がないという結果自体は研究上記録するが、hardware superiorityの主張は行わない。

λが小さい場合は物理的背景と正値応答を確認しlog-domainで計算する。数値floorを10倍変えた際の目的/勾配変化が数値誤差budgetを超えればFAIL。Poisson標本を直接pathwise backwardしない。主設計は期待Fisher、標本は独立評価とする。

Done：hard設計、独立holdout、複数初期値、公差/背景変動、固定MLEMの結果が揃い、改善/非改善の主張がCIで裏付けられる。

## 4. Phase 4 — スケーラビリティと学術/OSS

### 4.1 メモリと計算量

全チャネル組合せやE×V行列を構築しない。Eを観測数、Bをchunk数、Qt/Qperpを縦/横求積数、Hを経路仮説数とすると、主要な評価量は

$$O(E H Q_t Q_\perp C_{phys}),$$

一方working memoryは概ね

$$O(B H Q_tQ_\perp C_{state}+V+N_{channel}+K^2)$$

とし、Eに比例する計算グラフを保持しない。Kは少数の病変/nuisanceパラメータ。Cphysには材料検索・減弱積分を含む。全検出器SDFを各点で総当りする実装を避け、保守的局所検索を行う。支持集合は幾何変更で更新し、新規に寄与する境界の勾配を落とさない。

Fisherはsmall-matrix集約を用いる。1パス目をno_gradでIを集約し、G=∂L/∂Iを求め、2パス目で各chunkのG:I_chunkをbackwardする。両パスは同じ幾何/求積/乱数を使用する。

実装順はevent chunking→画像tiling/連続補間→局所材料lookup→checkpoint/2-pass→mixed precision→compile→必要時custom CUDA/Triton。checkpointは再計算とmemoryの交換であり、forwardとbackwardで状態が変わると誤る [R6]。`use_reentrant=False`を明示し、採用環境のAPIでテストする。

FP16から始めない。FP64 referenceを維持し、採用FP32ではFisherとreductionを必要に応じFP64にする。custom kernel化はprofileで支配的kernelが特定された場合だけ行い、fへの随伴とθへのVJPを両方再検証する。unrolled/implicit再構成で高階微分が必要なら、対象演算の対応を別ゲートにする。

### 4.2 G4 benchmark ladder

チャネル数10^4/3×10^4/10^5、画像128^3/256^3、観測10^5/10^6/10^7を設定する。初期目標は24GB級GPUで256^3・10^5チャネル・10^7イベントをstreamし、peak reserved≤20GB。これは未測定のengineering target。

固定chunkでEを10倍にしてdevice peak増加≤10%、event方向のruntime log-log slope≤1.2を目標とする。host側もbounded streamingを確認する。全イベントをCPUメモリへ読み込んでGPUだけconstant-memoryと称さない。

高速化版はFisher差≤1%、非零幾何勾配のrelative L2差≤2%、cosine≥0.99、GPU adjoint≤1e-5を満たす。cold/warm、compile、I/O、CPU/GPU時間、allocated/reserved、seed、dtype、driverを記録。microbenchmarkは可能な場合30反復以上でmedian/p95、重いend-to-endは独立反復数とCIを示す。CUDA同期を含める。

### 4.3 必須アブレーションと公平な比較

| 比較 | 分離する論点 |
|---|---|
| 高精度voxel/subvoxel vs直接Tube vs条件付きTube | 座標縮約とGaussian近似の値/勾配bias |
| hard/reference geometry vs soft固定τ vs anneal | 境界勾配と緩和gap |
| 単一Gaussian vs Doppler mixture vs多重散乱対応 | SRFの中心幅と裾の寄与 |
| joint / PET-only / Compton-only | 同一予算での観測情報の寄与 |
| 正しい絶対感度 vs 感度を捨てたnegative control | 感度喪失による偽の最適設計 |
| 同じ解析モデルのAD vs FD vs BO/CMA-ES | 勾配自体の効用 |
| MC格子/直接探索 vs MCを評価器とするBO | MC予算と到達性能 |
| 解析-only vs MC anchor付きtrust region | モデル悪用への耐性 |

MC直接探索は1D/2Dの完走可能な領域を必ず含める。多変数は全手法に同じbudgetと可行領域を与える。equal evaluation countとequal wall-clockを併記し、最終候補を全て独立MCで再評価する。提案手法の校正/前処理費用を隠さず、償却前と複数設計時の費用を示す。

### 4.4 論文戦略

主張候補は、絶対計数を保持するvoxelwise joint operator、誤差制御された低次元積分と幾何勾配、未知geometryでの物理/勾配妥当性、独立MC/実測での設計判断、計算/再現性である。WGI自体や交点画像化には先行研究がある [R1]。「世界初」「完全な任意トポロジーへの外挿」を検証なしに主張しない。

構成案：背景/既存研究→観測・測度契約→数理/ソフトウェア→事前登録検証設計→G1数値→G2物理/微分→G3設計/病変→G4計算/ablation→誤差budget/限界。

TMI向けには一般化可能な計算手法と独立タスク検証、Medical Physics向けには物理応答・不確実性・ファントム実証を前面に出すという投稿戦略を提案する。投稿先の正式scope/規定は投稿時に再確認する [R7,R8]。NSS/MICには到達済みG1/G2と限定G3を段階報告できるが、未実施の結果を予告値で埋めない。

実測が利用できる場合、モジュールenergy/ARM SRFと2以上の相対配置、または初期/最終相当のファントム測定を独立検証に用いる。実測がない場合はsimulation-onlyへ主張を限定し、実機/臨床的有効性を主張しない。採択そのものではなく、査読で検証可能な証拠パッケージの完成をG4とする。

### 4.5 OSS成果物

Python package、API/単位契約、small fixtures、MC generation scripts、評価CLI、CPU unit CI、GPU regression、MC golden-data integration、失敗ケース、version lock、container digestを用意する。全図をdata/config/code hashから再生成する。大容量MCは権限のあるarchiveに置き、repoはmanifest/hashを保持する。

OSSライセンス、データ再配布、依存ライセンス、著者/CITATION.cff、archive DOIはリリース前に確認する。現時点では未選択/未発行。公開repoへ患者情報や非公開協力機関データを登録しない。

## 5. すぐ着手するための配置と環境

以下は実装先の設計であり、現時点で存在するPythonモジュールではない。

```text
src/torch_wgi/
  geometry/{parameters,sdf,readout}.py
  physics/{transport,decay,energy,doppler,paths}.py
  operators/{reference_joint,tube,conditioned_tube,adjoint}.py
  reconstruction/listmode_mlem.py
  tasks/{fisher,observers}.py
  validation/{gradients,adjoint,statistics,mc_compare}.py
  io/{events,manifest}.py
configs/{geometry,physics,phantoms,experiments}/
tests/{unit,operator,integration}/
mc/{geometry_export,run_reference,event_extract}.py
benchmarks/{operator,memory,optimization}.py
paper/{figures,tables}/
```

最初のPRは`pyproject.toml`、単位/shapeを持つEventBatch/Geometry、平滑phantom、ReferenceJointProjector、勾配/随伴テストに限定する。次のPRでSDF/Soft-binning、続いて条件付きTube、最後にG1の全レポート。CI scaffoldingは初期から入れ、custom CUDAや多重散乱は最初のPRへ詰め込まない。

CPU toyのbootstrap例（研究結果はこの未固定環境で公開せず、動作確認後にバージョン固定する）：

```bash
python -m venv .venv
# POSIX shell; Windowsでは対応するactivateコマンドを使用
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy scipy pytest pyyaml
python -m pip freeze > requirements.bootstrap.lock
```

MCは依存衝突を避けて別環境/コンテナにし、OpenGATEの公式導入手順と互換Geant4データを使用する。PyTorch/CUDA/GATEの具体的版は、環境で実際に動いた組合せをlockfileとcontainer digestに固定する。「latest」を再現性の指定にしない。

実装後に提供するCLI契約例（現時点では実行不能）：

```text
python -m torch_wgi.validation.run --suite phase1 --config ...
python -m torch_wgi.validation.mc_compare --manifest ...
python -m torch_wgi.optimize --config ...
python -m torch_wgi.reconstruct --events ... --geometry ...
```

各experimentには `configs/experiment-manifest.example.yaml` の項目を保存する。各ゲートは `configs/validation-gates.yaml` のIDで参照し、PRに結果・CI・データhashを紐付ける。NOT RUN/FAIL/INCONCLUSIVEをPASSへ置換するのは証拠が揃った時だけとする。

## 6. 参照文献・公式資料

以下はモデル/実装判断の背景資料であり、本ロードマップ独自の数値閾値の出典ではない。参照確認日：2026-10-06。

- [R1] Yoshida et al. (2020), Whole gamma imaging: a new concept of PET combined with Compton imaging. Physics in Medicine & Biology 65, 125013. DOI: 10.1088/1361-6560/ab8e89. https://epub.ub.uni-muenchen.de/89331/
- [R2] PyTorch, Gradcheck mechanics. https://docs.pytorch.org/docs/stable/notes/gradcheck.html
- [R3] Geant4 Physics Reference Manual, Compton scattering / atomic shell effects. https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html
- [R4] OpenGATE, How to: Physics. https://opengate-python.readthedocs.io/en/master/user_guide/user_guide_physics.html
- [R5] Barrett, White, Parra (1997), List-mode likelihood. DOI: 10.1364/JOSAA.14.002914. https://pubmed.ncbi.nlm.nih.gov/9379247/
- [R6] PyTorch, torch.utils.checkpoint. https://docs.pytorch.org/docs/stable/checkpoint.html
- [R7] IEEE Transactions on Medical Imaging, official site. https://ieeetmi.org/
- [R8] Medical Physics, publisher's journal page. https://aapm.onlinelibrary.wiley.com/journal/24734209
