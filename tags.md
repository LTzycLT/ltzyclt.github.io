---
layout: default
title: "Tags"
permalink: /tags/
author_profile: true
---

<div id="main" role="main">
  {% include sidebar.html %}

  <div class="archive-content">
    <article class="archive">
      <h1 class="page__title">{{ page.title }}</h1>

      <div class="tag-cloud">
        {% for tag in site.tags %}
          <a href="/tags/{{ tag[0] | slugify }}/" class="tag-link">{{ tag[0] }}</a>
        {% endfor %}
      </div>
    </article>
  </div>

  <aside class="sidebar__right">
    <h3>Tags</h3>
    <nav class="sidebar-nav">
      <ul>
        {% for tag in site.tags %}
          <li><a href="/tags/{{ tag[0] | slugify }}/">{{ tag[0] }}</a></li>
        {% endfor %}
      </ul>
    </nav>
  </aside>
</div>