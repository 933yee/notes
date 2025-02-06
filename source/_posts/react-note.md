---
title: React 筆記
date: 2025-02-05 22:21:19
tags: [frontend, react, web]
category:
---

> 記錄一下常用到的 Components

### Swiper

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

### Collapse

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

### Select

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
