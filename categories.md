---
layout: default
title: "Categories"
permalink: /categories/
author_profile: true
---

<div id="main" role="main">
  {% include sidebar.html %}

  <div class="archive-content">
    <article class="archive">
      <h1 class="page__title">{{ page.title }}</h1>

      <ul class="category-list">
        {% for category in site.categories %}
          <li><a href="/categories/{{ category[0] | slugify }}/">{{ category[0] }}</a> ({{ category[1].size }})</li>
        {% endfor %}
      </ul>
    </article>
  </div>

  <aside class="sidebar__right">
    <h3>Categories</h3>
    <nav class="sidebar-nav">
      <ul>
        {% for category in site.categories %}
          <li><a href="/categories/{{ category[0] | slugify }}/">{{ category[0] }}</a></li>
        {% endfor %}
      </ul>
    </nav>
  </aside>
</div>