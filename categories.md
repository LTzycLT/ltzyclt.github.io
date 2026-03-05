---
layout: default
title: "Categories"
permalink: /categories/
author_profile: true
---

<div id="main" role="main">
  {% include sidebar.html %}

  <div class="archive">
    <h1 class="page__title">{{ page.title }}</h1>

    {% capture site_categories %}{% for category in site.categories %}{{ category[0] }},{% endfor %}{% endcapture %}
    {% assign sorted_categories = site_categories | split: ',' | sort %}

    <ul class="category-list">
      {% for category in sorted_categories %}
        {% if category != "" %}
          <li><a href="/categories/{{ category | slugify }}/">{{ category }}</a> ({{ site.categories[category].size }})</li>
        {% endif %}
      {% endfor %}
    </ul>
  </div>
</div>