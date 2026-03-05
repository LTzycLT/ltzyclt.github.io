---
layout: default
title: "Tags"
permalink: /tags/
author_profile: true
---

<div id="main" role="main">
  {% include sidebar.html %}

  <div class="archive">
    <h1 class="page__title">{{ page.title }}</h1>

    {% capture site_tags %}{% for tag in site.tags %}{{ tag[0] }},{% endfor %}{% endcapture %}
    {% assign sorted_tags = site_tags | split: ',' | sort %}

    <div class="tag-cloud">
      {% for tag in sorted_tags %}
        {% if tag != "" %}
          <a href="/tags/{{ tag | slugify }}/" class="tag-link">{{ tag }}</a>
        {% endif %}
      {% endfor %}
    </div>
  </div>
</div>