# 阅读指南 B：详情演示与手势识别线（Detail, Demonstration & Recognition）

对应用户旅程：**"我点进一个词，看它怎么比划、标记学会、以后还能跟着识别练习"** —— 从点击一张卡片，到看视频演示、切换标签跳分类、标记已学，以及为未来"手势识别"核心功能预留的接口。

负责这条线的人要能独立讲清楚："用户点进详情页之后，看到的每一块东西是怎么来的，以及'手势识别'这个核心功能现在到什么程度、以后怎么接上"。

---

## 0. 先看共同地基（两条线都要懂，~10分钟）

在看自己这条线之前，先花几分钟看这三个文件，不然后面的代码看不懂上下文：

| 文件 | 行数 | 要看懂什么 |
|---|---|---|
| `website/client/src/api/types.ts` | 40 | `Sign`/`SignVideo`这两个类型长什么样——尤其`videos`和`previewVideo`的区别 |
| `website/server/src/db.js` + `website/server/src/index.js` | 39 | 服务怎么启动、怎么连数据库、路由怎么挂载 |
| `website/server/schema.sql` | 73 | `sign_videos`表的字段设计，重点看`sign_id`外键和`source_id`唯一约束——为什么一个sign能挂多个视频 |

---

## 1. 推荐阅读顺序（按用户点击后的先后顺序看）

1. `server/src/utils/video.js` — 全文16行都要看
2. `routes/signs.js` 第122-155行（`GET /:id` handler，文件总共157行，只看这一段，前面0-120行是另一条线的`GET /`和`GET /tags`，不用管）
3. `api/signs.ts` 第31-35行（`fetchSignById`函数）
4. `pages/SignDetailPage.tsx` — 全文114行都要看，重点是第18-32行（数据获取的`useEffect`）和第87-93行（"Mark as learned"按钮）
5. `components/SignDemonstration.tsx` — 全文122行都要看，重点是第16-30行（过滤可用视频+两个`useEffect`）和第45-59行（`<video>`元素）
6. `lib/learnedSigns.ts` — 全文34行都要看
7. `components/Pagination/HandGlyphPagination.tsx` 全文71行（重点第9-25行`getVisiblePages`） + `handGlyphs.tsx` 只看第12-29行（`base()`辅助函数 + 第一个`HandGlyph1`，看懂这18行其余8个函数结构完全一样可以跳过），第99-112行（`HAND_GLYPHS`数组和`glyphForIndex`），第114行往后的"两只手计数"部分（第120-270行）不必细看，知道存在即可
8. `server/src/routes/recognize.js` — 全文52行都要看，核心是第32-50行的响应体结构

---

## 2. 逐文件讲解

### 后端

**`server/src/utils/video.js`（第1-16行，全文件）**
```js
const VIDEO_BASE_URL = (process.env.VIDEO_BASE_URL || "").replace(/\/+$/, "");
function formatVideo(row) {
  return {
    sourceId: row.source_id,
    fileName: row.file_name,
    videoUrl: row.video_url || (VIDEO_BASE_URL ? `${VIDEO_BASE_URL}/${encodeURIComponent(row.file_name)}` : null),
  };
}
```
**要点**：数据库`sign_videos.video_url`这一列可以是空的——如果DS团队只知道视频文件名（比如从Cloudflare R2批量上传后拿到一堆`.mp4`文件名），不用手动拼完整URL存进数据库，后端会自动用环境变量`VIDEO_BASE_URL`（指向R2的公开CDN域名）+ `encodeURIComponent(文件名)`拼出完整地址。`row.video_url ||`表示：如果数据库里已经手动填了完整URL就优先用那个，没填才自动拼。`replace(/\/+$/, "")`是去掉环境变量末尾可能多余的斜杠，防止拼出`.../videos//xxx.mp4`这种双斜杠。**视频文件本身从不经过我们的服务器和数据库，数据库只存文件名/URL指针，真正的文件在Cloudflare R2上——这样数据库体积小、服务器不用处理大文件流量。**

**`server/src/routes/signs.js` 第122-155行 —— `GET /api/signs/:id`**（文件总共157行，只看这34行，前面122行属于另一条线）
```js
const [rows] = await pool.query("SELECT * FROM signs WHERE id = ?", [id]);
if (rows.length === 0) return res.status(404).json({ error: "Sign not found", id });

const [videoRows] = await pool.query(
  `SELECT source_id, file_name, video_url FROM sign_videos WHERE sign_id = ? ORDER BY source_id ASC`, [id]
);
const sign = formatSign(rows[0]);
sign.videos = videoRows.map(formatVideo);
res.json(sign);
```
先查这个sign本身，查不到就`404`；再查它挂的所有视频（一个sign可能有多个视频版本，比如不同来源/不同拍摄角度），`ORDER BY source_id`保证顺序稳定。**和列表页`GET /`的区别**：列表页每张卡片只批量取"第一个"视频当预览（性能优先），详情页是单条查询，把这个sign的**所有**视频版本都返回，给用户切换着看。`id`要先`Number()`转换再判断`Number.isInteger`，防止SQL注入或者传"abc"这种非法id导致后端报错。

### 前端

**`api/signs.ts` 第31-35行 —— `fetchSignById`**
和另一条线的`fetchSigns`同一个套路（拼URL、fetch、检查`res.ok`、`return res.json()`），只是路径变成`/api/signs/${id}`。这个文件只有这5行是你这条线的，其余`fetchSigns`/`fetchTags`属于另一条线。

**`pages/SignDetailPage.tsx`（第1-114行，全文件）**
第18-32行：
```ts
useEffect(() => {
  if (!Number.isInteger(signId)) return;
  const controller = new AbortController();
  setSign(null); setIsError(false);
  fetchSignById(signId, controller.signal)
    .then((s) => { setSign(s); setLearned(isLearned(s.id)); })
    .catch((err) => { if (err.name !== "AbortError") setIsError(true); });
  return () => controller.abort();
}, [signId]);
```
标准的"数据获取三态"模式：`sign===null`时显示"Loading…"，`isError`时显示错误+返回按钮，都拿到了才渲染正文。拿到数据后立刻用`isLearned(s.id)`（另一个文件里的函数）查一下这个词是不是已经被标记学过，同步到本地`learned` state。`useEffect`依赖`[signId]`——路由参数变了（比如从详情页1跳到详情页2）会重新拉取。

页面结构（第53-113行的JSX，CSS Grid布局，样式在`theme.css`里，你只需要认识这几块）：
- 第56-61行 `.detail-back`：返回按钮 + 编号（`navigate(-1)`是浏览器历史回退，**这就是"从tag搜索结果点进详情页，返回后还留在搜索状态"需求的实现**——因为是真的"后退一步"而不是"跳转回首页"，URL上的`?query=`/`?tag=`参数会原样保留）
- 第63-65行 `.detail-media`：内嵌`<SignDemonstration>`播放器
- 第67-94行 `.detail-title`：词条名、来源、tag chips（第72-85行，点击会带着`?tag=xxx`跳回首页——这是"点tag看该分类下所有词"需求的实现，和另一条线的`CategoryRail`复用同一套URL约定）、第87-93行"Mark as learned"按钮
- 第96-110行 `.detail-definitions-area`：释义列表

**`components/SignDemonstration.tsx`（第1-122行，全文件，这条线最复杂的组件，重点看）**

第16-30行：
```ts
const availableVideos = videos.filter((video) => Boolean(video.videoUrl));
const currentVideo = availableVideos[videoIndex];

useEffect(() => {
  if (videoIndex >= availableVideos.length) setVideoIndex(0);
}, [availableVideos.length, videoIndex]);

useEffect(() => {
  if (videoRef.current) videoRef.current.playbackRate = speed;
}, [speed, videoIndex]);
```
- 先过滤掉`videoUrl`是`null`的视频（比如数据库里有记录但R2上文件还没传/URL没配好的脏数据），`availableVideos.length===0`时显示"暂无演示视频"文案，而不是渲染一个播放不了的空播放器。
- 第一个`useEffect`是防御性代码：如果用户正在看第3个视频版本，但这个sign只有2个可用视频了（理论上运行时不会变，但防止`videoIndex`越界导致`currentVideo`是`undefined`）。
- 第二个`useEffect`：切换播放速度时，通过`videoRef.current.playbackRate`直接操作DOM视频元素（React没有能直接控制这个属性的声明式API，只能用ref命令式设置）。

第45-59行：
```tsx
<video ref={videoRef} key={currentVideo.videoUrl ?? currentVideo.fileName} controls playsInline preload="metadata" className="sign-demo-video">
  <source src={currentVideo.videoUrl ?? undefined} type="video/mp4" />
</video>
```
`key={currentVideo.videoUrl}`很关键：切换视频版本时，`key`变化会让React**整个重新挂载**这个`<video>`元素而不是复用DOM节点，避免"切换了视频源但浏览器还在播放旧视频缓存帧"的显示bug。

播放控制区（第62-79行`video-variants`版本切换按钮组，只在这个sign有多个视频版本时才显示；第81-119行`playback-controls`）：Replay按钮（第86-95行）直接操作`videoRef.current.currentTime = 0; videoRef.current.play()`；Speed下拉框（第97-118行）改`speed` state，同步设置`playbackRate`。

**`lib/learnedSigns.ts`（第1-34行，全文件）**
```ts
const KEY = "auslan-website.learnedSigns.v1";
function readIds(): number[] { ...从localStorage读JSON数组，try/catch防止解析失败或用户手动改坏了数据... }
function writeIds(ids: number[]) { window.localStorage.setItem(KEY, JSON.stringify(ids)); }
export function isLearned(id: number): boolean { return readIds().includes(id); }
export function toggleLearned(id: number): boolean {
  const ids = readIds();
  const index = ids.indexOf(id);
  if (index === -1) { ids.push(id); writeIds(ids); return true; }
  ids.splice(index, 1); writeIds(ids); return false;
}
```
**这是全项目里唯一"写数据"的功能，而且是刻意不进数据库的**。整个"已学标记"就是往`localStorage`存一个纯数字id数组，`toggleLearned`每次都重新读一遍再写回去（没有做成全局state/context，因为一次只操作一个id，重新读写的开销可以忽略，没必要为这么小的功能引入状态管理）。**必须能讲清楚为什么不进数据库**：项目现在没有账号/登录系统，"这个用户学没学过某个词"是纯粹的个人本地数据，没有"用户"这个概念可以关联到数据库里，也没必要为了这一个小功能去设计整套账号体系。

**`components/Pagination/HandGlyphPagination.tsx`（第1-71行，全文件）+ `handGlyphs.tsx`（只看第1-29行和第99-112行）**
分页组件，UI风格要求保留不做改动。`HandGlyphPagination.tsx`第9-25行`getVisiblePages`函数处理"页数很多时怎么省略中间页码"：总页数≤8全部显示；超过8页时，只保留第一页、最后一页、当前页、当前页前后各一页，中间用`...`折叠，这是常见的分页UI算法（类似Google搜索结果的翻页样式）。`handGlyphs.tsx`第12-29行是`base()`样式辅助函数 + 第一个`HandGlyph1`，这个文件里一共有9个`HandGlyphN`函数（第22-97行）是纯装饰性的抽象"手掌+手指"线条图标（**注释里写明了：故意设计成不是真实的Auslan手形/指拼字母，纯装饰用，避免被误认成教学内容**），每个函数结构完全一样（`<svg>`+固定的`base()`样式+不同的`<path>`坐标），看懂第一个（第22-29行）就等于看懂全部，第30-97行不用逐个看。第99-112行`HAND_GLYPHS`数组和`glyphForIndex`也要看一眼。第114行往后（到270行）是"翻页数字用数手指方式显示"的另一套图标（`HandCount1-5`、双手计数），量大但同理是重复模式，答辩时知道它存在、说得出用途即可，不强制逐行看。

### 手势识别接口（核心功能，重点准备）

**`server/src/routes/recognize.js`（第1-52行，全文件；第5-31行是注释里写的接口契约，第32-50行是实际返回逻辑）**
```js
router.post("/", (req, res) => {
  res.status(501).json({
    status: "not_implemented",
    message: "Sign recognition is not implemented yet -- this endpoint reserves the contract for a future AI model.",
    expectedRequestShape: {
      targetSignId: "number",
      landmarks: 'Array<{timestamp:number, hand:"left"|"right", points:Array<{x:number,y:number,z:number}>}>',
    },
    expectedResponseShapeWhenImplemented: {
      status: "ok", targetSignId: "number", predictedLabel: "string", confidence: "number (0-1)", isMatch: "boolean",
    },
  });
});
```
**诚实地讲这个功能现在的状态**：这不是"假装做了"，而是明确返回`501 Not Implemented`加一份完整的接口契约文档。**为什么要先定契约再等以后实现**：这样前端/其他团队成员现在就能按这个格式对接（比如先写好调用这个接口、处理loading/结果展示的UI），不用等模型真正训练好才能开始联调；以后接AI识别模型时，也不需要因为接口设计临时改动前端代码。

**关键设计**：`landmarks`（手部关键点坐标）预期是**前端在浏览器里用MediaPipe提取好**再传给后端，**不是把视频文件传上来**——这样服务器不需要处理视频流/做视频解码，只需要处理一组坐标数字，性能开销小很多，也保护了用户隐私（原始摄像头画面不需要离开用户的设备）。仓库里的`web/`目录是这个方案的原型验证（如果时间够可以简单看一眼，不算在这条线的必读范围）。

答辩时如果被问"识别功能做到哪一步了"，可以按这个逻辑回答：**接口契约已经设计完成并预留好位置，前端采集方案（浏览器端MediaPipe提取关键点）已经验证可行，模型训练/比对算法是下一阶段的工作**——不要说"还没做"，要说"接口和数据流已经设计定型，只差接入具体模型"。

---

## 3. 一次完整请求走一遍（背下来，答辩时直接讲这个）

场景：用户从"greeting"分类的搜索结果里点开"HELLO"这个词。

1. `ResultCard`是个`<Link to="/signs/3">`，点击后React Router导航到`/signs/3`
2. `SignDetailPage`挂载，`useParams`拿到`id="3"`，调用`fetchSignById(3)`
3. 前端发出 `GET /api/signs/3`
4. 后端查`signs`表拿到这一行，再查`sign_videos`表拿到这个sign挂的所有视频（可能有2个版本），`formatVideo`把每个视频的`file_name`拼成完整R2地址
5. 前端拿到`sign`对象，渲染标题、tag chips、`<SignDemonstration videos={sign.videos}>`
6. `SignDemonstration`过滤出`videoUrl`不为空的视频，默认播放第一个，用户可以点"Version 2"切到第二个版本，或调速度
7. 用户点"Mark as learned" → `toggleLearned(3)`把id=3写进`localStorage`，按钮立刻变成"✓ Learned"样式，且下次回到列表页时`ResultCard`会读到这个标记显示徽章
8. 用户点某个tag chip（比如"greeting"）→ 跳转到`/?tag=greeting` → 回到浏览查找那条线的逻辑
9. 用户点"Back to library" → `navigate(-1)`是浏览器历史回退，不是跳回首页——如果是从`/?tag=greeting&page=2`点进来的，退回去还是`/?tag=greeting&page=2`，搜索状态完整保留

---

## 4. 可能被问到的问题

- **Q：视频为什么不直接存数据库？**
  A：视频是大文件，存数据库（BLOB）会让数据库体积暴涨、备份/迁移都变慢，而且MySQL不是为流式传输大文件设计的。存在Cloudflare R2这种对象存储+CDN上，数据库只存一个文件名/URL指针，查询和备份都轻量，用户访问视频时也是直接从CDN边缘节点拉取，比从我们自己的服务器中转要快。

- **Q："已学"状态为什么不做进数据库，以后想跨设备同步怎么办？**
  A：现在项目没有账号系统，`localStorage`是唯一能低成本实现"个性化状态"的地方。如果以后加账号系统，这部分完全可以平迁到"用户表+已学记录表"，不影响现在其他任何功能——这是一个刻意的、可退可进的最小实现，不是遗留漏洞。

- **Q：手势识别为什么现在只是个桩接口，进度是不是太慢？**
  A：识别模型训练/选型是DS方向的工作，需要单独的时间线；工程侧提前把接口契约、数据传输方案（浏览器端提取关键点而不传视频）设计定型，是为了让两边可以并行推进——DS训练模型的同时，前端已经能把"怎么采集用户手势、怎么展示识别结果"这部分UI先做出来，模型训练完直接接上，不需要返工接口。

- **Q：一个sign为什么允许挂多个视频？**
  A：同一个手语词可能有不同来源/不同签署者的示范视频（比如`sign_videos.source_id`标识来源），展示多个版本能让学习者看到同一个手势的自然变体，避免误以为"只有一种比法是对的"。
