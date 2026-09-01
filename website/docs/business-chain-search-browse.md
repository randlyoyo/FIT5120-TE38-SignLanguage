# 学习笔记：搜索与浏览线（Search & Browse）完整业务链条

> 用途：零基础自学笔记 / 答辩讲稿骨架
> 对应用户旅程："我想找一个手语词" —— 从打开首页、打字搜索/点分类，到看到结果列表、翻页。
> 配套阅读：`reading-guide-search-browse.md`（逐文件代码讲解）

---

## 一句话总结

> **用户在搜索框打字 / 点分类 → 前端防抖后同步 URL → 发请求 → 后端在数据库里"模糊搜 + 精确筛 + 相关度排序 + 分页"，并批量带上预览视频 → 返回 JSON → 前端渲染成卡片网格 + 分页。**

---

## 全景图

```mermaid
graph TD
    subgraph 用户操作
        A1[在搜索框打字<br/>e.g. "hello"]
        A2[点左侧分类<br/>e.g. "greeting"]
        A3[点第 2 页]
    end

    subgraph 前端 client
        B[queryInput 本地state<br/>每键实时显示]
        C[useDebouncedValue 防抖 300ms<br/>hooks/useDebouncedValue.ts]
        D[同步进 URL ?query=hello&tag=greeting<br/>SignLibraryPage.tsx]
        E[useSignSearch 三态<br/>hooks/useSignSearch.ts]
        F[fetchSigns 发请求<br/>api/signs.ts]
    end

    subgraph 后端 server
        G[GET /api/signs<br/>routes/signs.js]
        H[parsePagination 解析页码<br/>utils/pagination.js]
        I[拼 WHERE<br/>LIKE + JSON_SEARCH 模糊搜<br/>JSON_CONTAINS 精确筛]
        J[COUNT 数总数<br/>CASE WHEN 相关度排序]
        K[LIMIT/OFFSET 取一页<br/>批量 IN 查预览视频]
        L[formatSign 翻译驼峰<br/>返回 SignsResponse]
    end

    subgraph 数据库
        M[(signs 表 + sign_videos 表)]
    end

    A1 --> B --> C --> D --> E --> F --> G
    A2 --> D
    A3 --> D
    G --> H --> I --> M
    I --> J --> M
    J --> K --> M
    K --> L
    L -->|JSON| F
    E -->|三态| N[渲染: 骨架屏/结果/错误/空状态<br/>+ HandGlyphPagination]
```

---

## 按用户旅程逐步走（答辩直接讲这段）

**场景**：用户搜 "hello"，已点 "greeting" 分类，翻到第 2 页。

### 📍 第 1 站｜打字（前端）
- 每敲一个字母，`queryInput`（本地 state）更新 → **输入框实时显示**，手感流畅
- 代码：`SignLibraryPage.tsx` → `const [queryInput, setQueryInput] = useState(...)`

### 📍 第 2 站｜防抖（前端）
- 停止输入 300ms 后，`useDebouncedValue` 才吐出稳定值 `debouncedQuery = "hello"`
- **为什么**：避免每敲一个字就发一个请求（请求风暴），也避免"打字时 URL 导航丢字符"的竞态
- 代码：`hooks/useDebouncedValue.ts`（定时器 + 清理函数）

### 📍 第 3 站｜同步 URL（前端）
- `useEffect` 发现 `debouncedQuery` 和 URL 里的 query 不同 → `setSearchParams` 把 URL 改成 `?query=hello&tag=greeting`（并删掉 page，重新搜索从第 1 页开始）
- **URL 是唯一真相来源**：刷新 / 分享链接状态不丢

### 📍 第 4 站｜发请求（前端）
- `useSignSearch` 监听到 `query/tag/page` 变化 → 调 `fetchSigns`
- 用 `AbortController` 取消上一次未完成的请求 → **防止竞态：旧结果覆盖新结果**
- 代码：`hooks/useSignSearch.ts` → `api/signs.ts` → 发出 `GET /api/signs?query=hello&tag=greeting&page=1`

### 📍 第 5 站｜后端解析（后端）
- `parsePagination` 把 `page/pageSize` 变成安全的数字和 `offset`（夹范围、防 NaN）
- 代码：`routes/signs.js` → `utils/pagination.js`

### 📍 第 6 站｜拼查询条件（后端，核心）
```sql
WHERE (gloss LIKE '%hello%'
   OR JSON_SEARCH(keywords,'one','%hello%') IS NOT NULL
   OR JSON_SEARCH(tags,'one','%hello%') IS NOT NULL
   OR JSON_SEARCH(definitions,'one','%hello%') IS NOT NULL)
  AND JSON_CONTAINS(tags, '"greeting"')
```
- **关键词 = 模糊**（LIKE + JSON_SEARCH），**分类 = 精确**（JSON_CONTAINS）
- 关键词条件之间 `OR`（任一命中），与分类条件之间 `AND`（同时满足）
- 全程用 `?` 占位符防 SQL 注入

### 📍 第 7 站｜计数 + 排序 + 取数（后端）
- `COUNT(*)` 数总数 → 算 `totalPages`
- `ORDER BY CASE WHEN` 按相关度排：**gloss命中(0) > keywords命中(1) > tags命中(2) > definitions命中(3)**，同级按字母序
- `LIMIT 6 OFFSET 6` 取第 2 页那 6 条

### 📍 第 8 站｜批量带预览视频（后端，防 N+1）
- 收集这一页所有 `sign.id` → **一条** `WHERE sign_id IN (...)` 查完 → 用 `Map` 分组，每个 sign 只取第一条当预览
- **避免对每条结果各查一次视频（N+1 反模式）**

### 📍 第 9 站｜返回（后端）
- `formatSign` 把下划线 `usage_notes` 翻译成驼峰 `usageNotes`，`tags ?? []` 兜底
- 返回 `SignsResponse`：`{ results, pagination, query }` —— 和前端 `types.ts` 完全对齐

### 📍 第 10 站｜渲染（前端）
- `isLoading` → 6 个 `ResultCardSkeleton`（骨架屏）
- `isError` → "Couldn't load..." 错误提示
- 结果为空 → `EmptyState`
- 有结果 → `ResultCard` 列表
- 底部 `HandGlyphPagination` 显示页码

### 📍 第 11 站｜翻页（回到第 4 站循环）
- 点第 2 页 → `updateParams({page:2})` → URL 变 `...&page=2` → `scrollTo(top:0)` → `useSignSearch` 重新走一遍，只是 page=2

---

## 两条分支路径

| 路径 | 触发 | 链路差异 |
|---|---|---|
| **搜索** | 打字 | 走 `query` 参数 + CASE WHEN 相关度排序 |
| **分类浏览** | 点 CategoryRail | 只走 `tag` 参数 + JSON_CONTAINS 精确筛选（无 query 时不排序，按字母序） |

分类栏数据来自 `GET /api/signs/tags`：把所有 `tags` 查出来 → `Map` 计数 → 返回 `[{tag, count}]`。**注意 `/tags` 路由必须写在 `/:id` 之前**，否则会被当成 id 吞掉。

---

## 相关代码文件索引

| 环节 | 文件 | 作用 |
|---|---|---|
| 数据契约 | `client/src/api/types.ts` | `Sign` / `SignsResponse` / `TagCount` 等类型 |
| 数据库结构 | `server/schema.sql` | `signs` 表 + `sign_videos` 表 |
| 数据库连接 | `server/src/db.js` | `createPool` 连接池 |
| 服务器入口 | `server/src/index.js` | 路由挂载、CORS、404/500 兜底 |
| 分页工具 | `server/src/utils/pagination.js` | `parsePagination` / `buildPaginationMeta` |
| 字段翻译 | `server/src/utils/formatSign.js` | 下划线 → 驼峰 |
| 核心路由 | `server/src/routes/signs.js` | `GET /`、`GET /tags`、`GET /:id` |
| API 封装 | `client/src/api/signs.ts` | `fetchSigns` / `fetchTags` / `fetchSignById` |
| 防抖 | `client/src/hooks/useDebouncedValue.ts` | 300ms 防抖 |
| 搜索三态 | `client/src/hooks/useSignSearch.ts` | data/loading/error + AbortController |
| 分类加载 | `client/src/hooks/useTags.ts` | 挂载时拉一次分类 |
| 页面组装 | `client/src/pages/SignLibraryPage.tsx` | 本地 state + URL 同步 + 渲染 |

---

## 答辩要点清单（12 个考点）

1. **tags vs keywords 为什么要分开**：展示用分类 vs 纯搜索同义词，合并会污染 UI / 混淆语义
2. **搜索模糊、筛选精确**：LIKE / JSON_SEARCH vs JSON_CONTAINS
3. **为什么 keywords 排在 tags 前面**：keywords 命中 = 精确命中用户意图，tags 只是顺带相关
4. **CASE WHEN 相关度排序**（而非引入 ES）：数据量小，SQL 足够，不过度设计
5. **防 N+1 查询**：收集 id → 批量 `IN` → `Map` 分组
6. **`/tags` 必须在 `/:id` 前**：Express 按注册顺序匹配
7. **为什么 tags 统计不用 GROUP BY**：JSON 数组列无法分组计数
8. **防抖 + URL 同步的竞态设计**：本地 state 保手感，防抖值才写 URL 才发请求
9. **AbortController 防竞态**：取消旧请求，防止旧结果覆盖新结果
10. **SQL 注入防御**：全程 `?` 占位符，绝不字符串拼接
11. **fetch 不自动抛错**：必须 `!res.ok` 手动判断
12. **默认 pageSize=6**：配合网格布局的视觉密度，非技术限制

---

## 进阶自测（答得上来才算真懂）

1. 用户输入 `?pageSize=999999`，`parsePagination` 返回的 pageSize 是多少？
   → 50（被 `Math.min(MAX_PAGE_SIZE, ...)` 夹住）
2. 每页 8 条，第 4 页的 offset 是多少？
   → (4-1)×8 = 24，跳过前 24 条
3. `buildPaginationMeta(1, 6, 35)` 的 totalPages 是多少？
   → 35÷6 ≈ 5.83 向上取整 = 6
4. 为什么搜索框不直接绑定 URL 参数？
   → `useSearchParams` 每次都是浏览器导航，打字快时导航竞态会丢字符
5. `fetch` 请求到 404 会自动抛异常吗？
   → 不会，必须 `if (!res.ok) throw ...` 手动判断
