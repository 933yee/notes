---
title: React 筆記
date: 2025-02-05 22:21:19
tags: [frontend, react, web]
category:
---

> 記錄一下常用到的 Components

# JSX

## shadcn

相見恨晚阿.. 太強了
[官方文件](https://ui.shadcn.com/docs)

## TanStack

官方有個非常詳盡的[文件](https://tanstack.com/table/latest)，這裡只有寫大概怎麼用。

文件寫到 `TanStack Table is a modular library. Not all code for every single feature is included in the createTable functions/hooks by default.`，所以很多功能要在建立 table 的時候自己 import。

- 安裝

  ```sh
  npm install @tanstack/react-table
  ```

### Header

Header Group 包含所有 header row

```jsx
<thead>
  {table.getHeaderGroups().map((headerGroup) => {
    return (
      <tr key={headerGroup.id}>
        {headerGroup.headers.map(
          (
            header // map over the headerGroup headers array
          ) => (
            <th key={header.id} colSpan={header.colSpan}>
              {/* */}
            </th>
          )
        )}
      </tr>
    );
  })}
</thead>
```

### Column

#### 可以從 `cell` 、 `header` 和 `table` 取得

```jsx
const column = cell.column; // get column from cell
const column = header.column; // get column from header
const column = table.getColumn("firstName");
```

####

### Row

全部可以用的 row model

```jsx
const table = useReactTable({
  columns,
  data,
  getCoreRowModel: getCoreRowModel(),
  getExpandedRowModel: getExpandedRowModel(),
  getFacetedMinMaxValues: getFacetedMinMaxValues(),
  getFacetedRowModel: getFacetedRowModel(),
  getFacetedUniqueValues: getFacetedUniqueValues(),
  getFilteredRowModel: getFilteredRowModel(),
  getGroupedRowModel: getGroupedRowModel(),
  getPaginationRowModel: getPaginationRowModel(),
  getSortedRowModel: getSortedRowModel(),
});
```

#### `table.getRow()`

```jsx
const row = table.getRow(rowId);
```

#### `table.getRowModel()`

```jsx
<tbody>
  {table.getRowModel().rows.map((row) => (
    <tr key={row.id}>{/* ... */}</tr>
  ))}
</tbody>
```

#### `table.getSelectedRowModel()`

```jsx
const selectedRows = table.getSelectedRowModel().rows;
```

#### `row.getValue()`、`row.renderValue()`

兩個只差在檢查 value 是否為 `undefined`

```js
const firstName = row.getValue("firstName"); // read the row value from the firstName column
const renderedLastName = row.renderValue("lastName"); // render the value from the lastName column
```

#### `row.original`

跟 `row.getValue()` 類似，但是回傳 `accessorFn` **處理前** 的值，是最原始的資料

```jsx
const firstName = row.original.firstName;
```

### Cell

每個 Cell 的 id 是由 parent row 和 parent col 決定的：`{ id: `${row.id}_${column.id}` }`

#### `cell.getValue()`、`cell.renderValue()`

```jsx
const firstName = cell.getValue(); // read the cell value
const renderedLastName = cell.renderValue(); // render the value
```

#### 取得 parent row

```jsx
const firstName = cell.row.original.firstName;
```

#### flexRender()

如果 `column.cell` 是 **函數回傳 JSX** 而不是純數據，就一定要用 `flexRender` 渲染，不能用 `getValue()`

```jsx
import { flexRender } from '@tanstack/react-table'

const columns = [
  {
    accessorKey: 'fullName',
    cell: ({ cell, row }) => {
      return <div><strong>{row.original.firstName}</strong> {row.original.lastName}</div>
    }
    //...
  }
]
//...
<tr>
  {row.getVisibleCells().map(cell => {
    return <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
  })}
</tr>
```

##### `getVisibleCells()`

只回傳當前可見的 Cells，避免 render 不需要的內容

```jsx
{
  table.getRowModel().rows.map((row) => (
    <tr key={row.id}>
      {row.getVisibleCells().map((cell) => (
        <td key={cell.id}>{cell.renderValue()}</td>
      ))}
    </tr>
  ));
}
```

## React Hook Form

透過 `useForm hook` 管理表單

- 安裝

  ```sh
  npm install react-hook-form
  ```

- `shadui` + `react hook form` + `zod` 驗證範例

  ```jsx
  import { useForm } from "react-hook-form";
  import { zodResolver } from "@hookform/resolvers/zod";
  import * as z from "zod";
  import { Button } from "@/components/ui/button";
  import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";
  import { Input } from "@/components/ui/input";
  import { Checkbox } from "@/components/ui/checkbox";

  const formSchema = z.object({
    name: z.string().min(2, "名稱至少需要 2 個字"),
    email: z.string().email("請輸入有效的 Email"),
    agreeTerms: z.boolean().refine(val => val === true, {
      message: "您必須同意條款與條件"
    })
  });

  export default function MyForm() {
    const form = useForm({
      resolver: zodResolver(formSchema),
      defaultValues: {
        name: "",
        email: "",
        agreeTerms: false
      }
    });

    const onSubmit = (data) => {
      console.log("Submitted data:", data);
    };

    return (
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 p-4 border rounded-lg w-full max-w-md">
          {/* Name Field */}
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormLabel>名稱</FormLabel>
                <FormControl>
                  <Input placeholder="輸入您的名稱" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Email Field */}
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input type="email" placeholder="輸入您的 Email" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Agree Terms Checkbox */}
          <FormField
            control={form.control}
            name="agreeTerms"
            render={({ field }) => (
              <FormItem className="flex items-center space-x-2">
                <FormControl>
                  <Checkbox checked={field.value} onCheckedChange={field.onChange} />
                </FormControl>
                <FormLabel>我同意條款與條件</FormLabel>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Submit Button */}
          <Button type="submit" className="w-full">提交</Button>
        </form>
      </Form>
    );
  }
  import { useForm } from "react-hook-form";
  import { zodResolver } from "@hookform/resolvers/zod";
  import * as z from "zod";
  import { Button } from "@/components/ui/button";
  import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";
  import { Input } from "@/components/ui/input";
  import { Checkbox } from "@/components/ui/checkbox";

  const formSchema = z.object({
    name: z.string().min(2, "名稱至少需要 2 個字"),
    email: z.string().email("請輸入有效的 Email"),
    agreeTerms: z.boolean().refine(val => val === true, {
      message: "您必須同意條款與條件"
    })
  });

  export default function MyForm() {
    const form = useForm({
      resolver: zodResolver(formSchema),
      defaultValues: {
        name: "",
        email: "",
        agreeTerms: false
      }
    });

    const onSubmit = (data) => {
      console.log("Submitted data:", data);
    };

    return (
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 p-4 border rounded-lg w-full max-w-md">
          {/* Name Field */}
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormLabel>名稱</FormLabel>
                <FormControl>
                  <Input placeholder="輸入您的名稱" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Email Field */}
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input type="email" placeholder="輸入您的 Email" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Agree Terms Checkbox */}
          <FormField
            control={form.control}
            name="agreeTerms"
            render={({ field }) => (
              <FormItem className="flex items-center space-x-2">
                <FormControl>
                  <Checkbox checked={field.value} onCheckedChange={field.onChange} />
                </FormControl>
                <FormLabel>我同意條款與條件</FormLabel>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Submit Button */}
          <Button type="submit" className="w-full">提交</Button>
        </form>
      </Form>
    );
  }
  ```

## Swiper

- 安裝

  ```sh
  npm install swiper
  ```

- import

  - 一定要的

    ```js
    import { Swiper, SwiperSlide } from "swiper/react";
    import "swiper/css";
    ```

  - 其他一些可用的 modules

    ```js
    import { EffectCoverflow, Pagination } from "swiper/modules";
    import "swiper/css/pagination";
    ```

- 用法範例
  - Swiper
    ```jsx
    <Swiper
      onSlideChange={(swiper) => setcurrentAppIntroIndex(swiper.activeIndex)}
      initialSlide={2}
      slidesPerView={"auto"}
      coverflowEffect={{
        rotate: 0,
        stretch: 80,
        depth: 300,
        modifier: 1,
        slideShadows: true,
      }}
      effect={"coverflow"}
      grabCursor={true}
      centeredSlides={true}
      onSwiper={(swiper) => (appSwiperRef.current = swiper)}
      pagination={{ el: ".app-pagination", clickable: true }}
      modules={[EffectCoverflow, Pagination]}
    >
      {appFeatures.map((item, index) => (
        <SwiperSlide key={index}>
          <div>{item.title}</div>
        </SwiperSlide>
      ))}
    </Swiper>
    ```
  - Pagination
    ```jsx
    <div
      className="app-pagination mt-2 flex justify-center space-x-4"
      style={{
        "--swiper-pagination-color": "#000000",
        "--swiper-pagination-bullet-inactive-color": "#999999",
        "--swiper-pagination-bullet-inactive-opacity": "1",
        "--swiper-pagination-bullet-size": "8px",
      }}
    />
    ```

## Collapse

- 安裝

  ```sh
  npm install --save react-collapse
  ```

- import

  ```jsx
  import { Collapse } from "react-collapse";
  ```

  另外，`index.html` 要加入

  ```html
  <script src="https://unpkg.com/react/umd/react.production.min.js"></script>
  <script src="https://unpkg.com/react-collapse/build/react-collapse.min.js"></script>
  ```

- 用法範例
  用 `isOpened` 控制 Toggle 就好，內容隨意

  ```jsx
  <Collapse isOpened={isExpanded[index]}>
    {question.description &&
      question.description.map((desc, index) => (
        <div>
          <div key={index}>{desc.description}</div>
          {desc.details && (
            <ul className="ml-10 list-disc">
              {desc.details.map((detail, index) => (
                <li key={index}>
                  <span className="font-bold">{detail.title}</span>
                  {detail.description}
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
  </Collapse>
  ```

  - Collapse 動畫的時間

  ```css
  .ReactCollapse--collapse {
    transition: height 500ms;
  }
  ```

## Select

- 安裝

  ```sh
  npm i --save react-select
  ```

- import

  ```jsx
  import Select from "react-select";
  ```

- 範例

  ```jsx
  const options = [
    { value: "chocolate", label: "Chocolate" },
    { value: "strawberry", label: "Strawberry" },
    { value: "vanilla", label: "Vanilla" },
  ];
  const MyComponent = () => <Select options={options} />;
  ```

- [更多用法](https://react-select.com/home#getting-started)

## Spinner

- 安裝

  ```sh
  npm install --save react-spinners
  ```

- import

  ```jsx
  import ClipLoader from "react-spinners/ClipLoader";
  ```

- 範例

  ```jsx
  <ClipLoader
    color="#ffffff"
    loading={loading} // bool 值
    size={150}
    aria-label="Loading Spinner"
    data-testid="loader"
  />

  <BeatLoader
    color="#000000"
    loading={true} // bool 值
    size={10}
    aria-label="Loading Spinner"
    data-testid="loader"
  />
  ```

- [更多用法](https://www.npmjs.com/package/react-spinners)

# CSS

## flexwrap

用 `flex` 的時候，會與下一行有一些奇怪的間隔，是因為他會啟動 `aligin-content: strench`，要改成 `align-content: flex-start`，或 Tailwind 的 `content-start`

- [更多資訊](https://stackoverflow.com/questions/40890613/remove-space-gaps-between-multiple-lines-of-flex-items-when-they-wrap)
