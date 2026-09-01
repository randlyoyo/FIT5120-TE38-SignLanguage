# 阅读指南 A：浏览与查找线（Search & Browse）

对应用户旅程：**"我想找一个手语词"** —— 从打开首页、打字搜索/点分类，到看到结果列表、翻页。

负责这条线的人要能独立讲清楚："用户在搜索框打一个字，到屏幕上出现结果，中间发生了什么"。

---

## 0. 先看共同地基（两条线都要懂，~10分钟）

在看自己这条线之前，先花几分钟看这三个文件，不然后面的代码看不懂上下文：

| 文件 | 行数 | 要看懂什么 |
|---|---|---|
| `website/client/src/api/types.ts` | 40 | `Sign`/`SignsResponse`/`TagCount`这几个类型长什么样——前后端的"数据契约" |
| `website/server/src/db.js` + `website/server/src/index.js` | 39 | 服务怎么启动、怎么连数据库、路由怎么挂载 |
| `website/server/schema.sql` | 73 | `signs`表和`sign_videos`表的字段设计，重点看`tags`字段和`keywords`字段的注释——为什么这俩要分开 |

---

## 1. 推荐阅读顺序（自底向上：先看数据从哪来，再看怎么展示）

1. `server/src/utils/pagination.js` 全文25行 → `server/src/utils/formatSign.js` 全文15行（后端小工具，一眼看完）
2. `routes/signs.js` 第17-99行（`GET /`）和第104-120行（`GET /tags`）两个handler（**这条线唯一有硬度的地方**，文件总共157行，第122-155行是`GET /:id`，属于另一条线，不用看）
3. `api/signs.ts` 第13-29行（`fetchSigns`）和第37-42行（`fetchTags`）
4. `hooks/useDebouncedValue.ts` 全文13行 → `useSignSearch.ts` 全文38行 → `useTags.ts` 全文21行（三个hook同一个模式，从最简单的看起）
5. `lib/tagColors.ts` 全文26行
6. `pages/SignLibraryPage.tsx` 全文132行（把上面所有东西拼起来的地方，重点第26-40行和第45-55行）
7. `components/CategoryRail.tsx`（全文37行）、`ResultCard.tsx`（全文66行）、`ResultCardSkeleton.tsx`（全文13行）、`SearchBar.tsx`（全文22行）、`EmptyState.tsx`（全文20行）、`PlaceholderMedia.tsx`（全文31行）——纯展示组件，随便看，都很短，每个都要通篇看完（没有"只看一部分"的情况，因为本身就很短）

---

## 2. 逐文件讲解

### 后端

**`server/src/utils/pagination.js`（第1-25行，全文件）**
```js
function parsePagination(query) {
  const page = Math.max(1, parseInt(query.page, 10) || 1);
  const pageSize = Math.min(MAX_PAGE_SIZE, Math.max(1, parseInt(query.pageSize, 10) || DEFAULT_PAGE_SIZE));
  const offset = (page - 1) * pageSize;
  return { page, pageSize, offset };
}
```
把`?page=&pageSize=`这两个query参数转成安全的`LIMIT/OFFSET`。要点：`parsePagination`函数（第5-13行）里`Math.min(MAX_PAGE_SIZE, ...)`防止有人传`pageSize=999999`把数据库查爆；`parseInt(...) || 默认值`防止传非法值（比如`page=abc`）导致`NaN`。`buildPaginationMeta`（第15-22行）则是反过来，算出`totalPages`给前端渲染页码用。默认`pageSize`是6（第1行`DEFAULT_PAGE_SIZE`，原来是4，按需求改大过）。

**`server/src/utils/formatSign.js`（第1-15行，全文件，核心函数在第2-12行）**
纯粹的字段映射：数据库返回的是`usage_notes`（下划线，MySQL习惯），API要吐出`usageNotes`（驼峰，JS习惯）。第9-10行`tags ?? []`和`keywords ?? []`是防止数据库里这两个JSON字段是`NULL`时前端拿到`undefined`报错。

**`server/src/routes/signs.js` 第17-99行 —— `GET /api/signs`（这是重点，务必吃透；文件总共157行，只看这段和下面的`GET /tags`）**

功能：关键词搜索 + tag筛选 + 排序 + 分页 + 批量拿预览视频。逐段拆：

第27-33行：
```js
if (trimmedQuery) {
  const like = `%${trimmedQuery}%`;
  conditions.push(
    "(gloss LIKE ? OR JSON_SEARCH(keywords, 'one', ?) IS NOT NULL OR JSON_SEARCH(tags, 'one', ?) IS NOT NULL OR JSON_SEARCH(definitions, 'one', ?) IS NOT NULL)"
  );
  params.push(like, like, like, like);
}
```
- `gloss LIKE '%xxx%'`：普通字符串模糊匹配（`gloss`是`VARCHAR`列）
- `JSON_SEARCH(列, 'one', 值)`：MySQL专门用来在JSON数组里模糊查值的函数，因为`keywords`/`tags`/`definitions`都是JSON类型，不能直接`LIKE`
- 四个条件用`OR`连——只要有一个字段命中就算匹配

第35-38行：
```js
if (trimmedTag) {
  conditions.push("JSON_CONTAINS(tags, JSON_QUOTE(?))");
  params.push(trimmedTag);
}
```
- 精确匹配一个tag（不是模糊搜索）。`JSON_QUOTE('greeting')`把字符串包成`"greeting"`，`JSON_CONTAINS`检查这个值是否在`tags`数组里。
- 和上面的`query`条件是`AND`关系（`conditions.join(" AND ")`）——可以同时"搜关键词+筛tag"。

第48-61行（`orderClause`赋值，CASE WHEN本体在第52-59行）：
```js
orderClause = `ORDER BY
  CASE
    WHEN gloss LIKE ? THEN 0
    WHEN JSON_SEARCH(keywords, 'one', ?) IS NOT NULL THEN 1
    WHEN JSON_SEARCH(tags, 'one', ?) IS NOT NULL THEN 2
    ELSE 3
  END ASC,
  gloss ASC`;
```
**这是需求"搜索顺序先keyword再tag"的实现**。用`CASE WHEN`给每一行打一个"相关度等级"：词条本身命中=0（最高优先级）、搜索关键词命中=1、tag命中=2、只有在definitions里出现=3（最低）。`ORDER BY ... ASC`让等级小的排前面，同等级再按字母序`gloss ASC`排。**要能讲清楚为什么keywords排在tags前面**：`keywords`是专门放"用户可能搜的同义词"的字段（比如HELLO配"hi","hey"），命中它就是精确命中用户意图；`tags`是分类标签，命中它只是"顺带相关"，相关度低一档。

第71-86行：
```js
const signIds = rows.map((row) => row.id);
const previewBySignId = new Map();
if (signIds.length) {
  const [videoRows] = await pool.query(
    `SELECT sign_id, source_id, file_name, video_url FROM sign_videos WHERE sign_id IN (?) ORDER BY sign_id ASC, source_id ASC`,
    [signIds]
  );
  for (const video of videoRows) {
    if (!previewBySignId.has(video.sign_id)) {
      previewBySignId.set(video.sign_id, formatVideo(video));
    }
  }
}
```
**要点：批量查，不是循环查。** 如果对每一页的6条结果各自查一次视频，就是6次SQL往返；这里先收集这一页所有`sign.id`，一条`WHERE sign_id IN (?)`查完，再用`Map`按`sign_id`分组，每个sign只取第一个视频（`if (!previewBySignId.has(...))`）作为列表页的预览。这是"N+1查询"反模式的标准修复手法，属于面试常考点，答辩时可以主动提。

**`server/src/routes/signs.js` 第104-120行 —— `GET /api/signs/tags`**
```js
router.get("/tags", async (req, res, next) => {
  const [rows] = await pool.query("SELECT tags FROM signs");
  const counts = new Map();
  for (const row of rows) {
    for (const tag of row.tags ?? []) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1);
    }
  }
  const tags = [...counts.entries()].map(([tag, count]) => ({ tag, count })).sort(...);
  res.json({ tags });
});
```
统计每个tag出现次数，给左侧分类导航栏用。**为什么不用SQL的`GROUP BY`统计**：`tags`是JSON数组列，一行可能同时属于`["family","greeting"]`两个tag，MySQL没法直接对JSON数组里的元素分组计数，所以退而求其次：把所有行的`tags`都查出来，在Node里用`Map`手动计数。**这个路由必须写在`GET /:id`之前**（代码里也有注释）：Express路由是按注册顺序匹配的，如果`/:id`先注册，访问`/api/signs/tags`会被当成`id="tags"`吃掉，走进详情页逻辑再报错。

### 前端

**`api/signs.ts` 第13-29行（`fetchSigns`）和第37-42行（`fetchTags`）**
纯HTTP封装：拼`URLSearchParams`、`fetch`、检查`res.ok`（HTTP状态码是不是2xx，`fetch`不会自动对4xx/5xx抛异常，必须手动判断）、`return res.json()`。文件里第31-35行`fetchSignById`同款套路（那个属于另一条线，不用看，但顺手认识一下也无妨）。

**`hooks/useDebouncedValue.ts`（第1-13行，全文件，先看这个，最简单）**
```ts
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}
```
经典防抖模式：每次`value`变化就设一个新定时器，如果在`delayMs`内`value`又变了，`useEffect`的清理函数会先清掉上一个定时器——所以只有"停止输入超过300ms"才会真正更新`debounced`。

**`hooks/useSignSearch.ts`（第1-38行，全文件，核心逻辑在第16-34行的`useEffect`）和 `useTags.ts`（第1-21行，全文件，看完上面这个，这两个就是同一个模式套壳）**
都是：`useState`存数据/loading/error三态 → `useEffect`里发请求 → `AbortController`在依赖变化或组件卸载时取消上一次没完成的请求（防止网络慢的时候，旧请求的结果比新请求后返回，把新结果覆盖掉——经典的竞态条件bug）。

**`lib/tagColors.ts`（第1-26行，全文件）**
把tag字符串哈希成一个固定色相（0-360度，第6-12行`hueForTag`），保证同一个tag无论出现在哪个组件里颜色都一致，不需要维护一张"tag→颜色"的映射表。

**`pages/SignLibraryPage.tsx`（第1-132行，全文件，这条线里前端最绕的一段，重点看第26-55行）**

第26-27行（本地state + 防抖）：
```ts
const [queryInput, setQueryInput] = useState(searchParams.get("query") ?? "");
const debouncedQuery = useDebouncedValue(queryInput, 300);
```
第29-40行（防抖后的值同步进URL）：
```ts
useEffect(() => {
  const currentQuery = searchParams.get("query") ?? "";
  if (debouncedQuery === currentQuery) return;
  setSearchParams((prev) => { ...把debouncedQuery写进URL... });
}, [debouncedQuery]);
```
第42行（真正触发请求）：
```ts
const { data, isLoading, isError } = useSignSearch({ query: debouncedQuery, tag, page });
```
**为什么搜索框不直接绑定URL参数**：`useSearchParams`背后是浏览器的history API，每次调用都是一次"导航"。如果每敲一个字符就调一次`setSearchParams`，打字快的时候后一次导航可能在前一次还没提交完就触发，会丢字符（这是真实踩过的坑）。所以设计成：搜索框自己管理本地state（每个按键都实时反映在输入框里，手感流畅）→ 防抖 300ms → 只有防抖后的值才同步进URL → 也只有这个防抖后的值才触发`useSignSearch`真正发请求。**这样输入体验流畅，同时避免了竞态和请求风暴。**

第45-55行`updateParams`里的`window.scrollTo({top:0,behavior:"smooth"})`（第54行）对应"翻页后回到顶部"的需求。第57-64行`clearTag`、第66-69行`clearFilters`是两个附带的清除逻辑，扫一眼即可。

**`components/CategoryRail.tsx`（第1-37行）/ `ResultCard.tsx`（第1-66行）/ `ResultCardSkeleton.tsx`（第1-13行）/ `SearchBar.tsx`（第1-22行）/ `EmptyState.tsx`（第1-20行）/ `PlaceholderMedia.tsx`（第1-31行）——都是全文件**
都是纯展示组件，没有自己的状态逻辑（`ResultCard.tsx`第39-41行读了一下`isLearned()`判断要不要显示徽章，这个函数属于另一条线，你只需要知道"它读了localStorage"即可）。`PlaceholderMedia`用`sign.id`算一个固定色相的渐变色块，在没有真实视频时当占位图，明确标注"Placeholder"字样，不冒充真实素材。

---

## 3. 一次完整请求走一遍（背下来，答辩时直接讲这个）

场景：用户在搜索框输入"hello"，同时已经点了"greeting"分类，在第2页。

1. 每敲一个字母，`queryInput`本地state更新，输入框实时显示
2. 停止输入300ms后，`useDebouncedValue`吐出新的`debouncedQuery="hello"`
3. `useEffect`发现`debouncedQuery`和URL里的`query`不一样，把URL改成`?query=hello&tag=greeting`（`page`参数被删掉，重新搜索从第1页开始）
4. `useSignSearch`监听到`query`变化，调用`fetchSigns({query:"hello", tag:"greeting", page:1})`
5. 前端发出 `GET /api/signs?query=hello&tag=greeting&page=1`
6. 后端`signs.js`：拼`WHERE (gloss LIKE '%hello%' OR ...) AND JSON_CONTAINS(tags, '"greeting"')` → 先`COUNT(*)`算总数 → 按"gloss命中>keywords命中>tags命中>definitions命中"排序取前6条 → 批量查这6条的预览视频 → 组装JSON
7. 前端拿到`results`渲染成`<ResultCard>`列表，`totalPages`交给`HandGlyphPagination`渲染页码（分页组件属于另一条线，你只需要知道它接收`page`/`totalPages`/`onPageChange`三个props）
8. 用户点第2页 → `updateParams({page:2})` → URL变成`...&page=2`并`scrollTo(top:0)` → 触发第4步重新走一遍，只是`page=2`

---

## 4. 可能被问到的问题

- **Q：为什么tags和keywords要分成两个字段，不合并？**
  A：tags是要展示给用户看的分类标签（UI上是彩色chip，也用于分类导航栏），keywords是纯粹给搜索用的同义词，从不展示。如果合并，要么污染UI显示一堆不该出现的同义词，要么让"分类"和"搜索联想词"这两个不同语义的东西混在一个字段里，查询和维护都会更麻烦。

- **Q：搜索是模糊匹配还是精确匹配？**
  A：`query`关键词搜索是模糊匹配（`LIKE '%xxx%'` + `JSON_SEARCH`），`tag`筛选是精确匹配（`JSON_CONTAINS`要求完全相等）——因为分类应该是明确的，不应该"搜greeting结果把greetings也带出来"这种模糊。

- **Q：为什么用CASE WHEN排序而不是搜索引擎（如Elasticsearch）？**
  A：当前数据量小（几十到几百条demo数据），SQL层面的简单相关度打分完全够用，没必要引入额外的搜索引擎基础设施，符合"不为假设中的未来需求过度设计"的原则。

- **Q：分页为什么默认6条而不是10条？**
  A：配合卡片网格`grid-template-columns: repeat(auto-fill, minmax(240px,1fr))`的视觉密度调整过，6条在常见屏宽下正好整齐排2~3行，是纯UI观感决定，不是技术限制（`MAX_PAGE_SIZE`是50，随时能改）。
