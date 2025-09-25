Updates about Recommender system and feed issues

We had a meeting with leadership and managers yesterday following was discussed:
- feed exhaustion is result of patch fixes we had done to support migration efforts
- immediate fix proposed and approved was: increase exploratory unwatched content for the user

What changes went live today (19 Sept 2025)
1. Fetch items from video corpus which have 0 interactions in last 90 days
2. We are calling it last_fallback it will return 50 additional items apart from existing fallbacks in place
3. This last_fallback is set to refresh everyday

GOAL:
- feed should not get exhausted

WHAT does this change NOT do:
- relevance of videos might go down
- as this change prioritizes exploratory content, relevance would naturally go down
- Please NOTE: the algorithm has been tweaked for power users, we do not have enough data points to recommend for these users and will need proper plan and design to make it relevant

These are some results of stress tests ran after this change:

Test Results Summary (19 Sept 2025)

We conducted stress tests on both NSFW and Clean content with 30+ requests per user session to simulate power user behavior:

NSFW Content Test Results:
- Content Discovery Rate: 93.34%
- Already Watched Rate: 6.66% (excellent freshness)
- Request Success Rate: 100% (312/312 requests)
- Average Videos Returned: 69.2 (130% more than requested 30)
- Content Shortage Rate: 0.00% (no feed exhaustion)

Clean Content Test Results:
- Content Discovery Rate: 93.84%
- Already Watched Rate: 6.16% (excellent freshness)
- Request Success Rate: 100% (316/316 requests)
- Average Videos Returned: 67.1 (124% more than requested 30)
- Content Shortage Rate: 0.63% (minimal shortage)

Key Performance Insights:
- 75% of requests had 0% already watched content - excellent early user experience
- 90% of requests had ≤35% already watched content - strong performance for most users
- Only 5% of requests had >50% already watched content - minimal impact on power users
- Zero feed exhaustion detected - goal achieved
- Response times: ~2.6-2.7 seconds - within acceptable range but room for optimization

Distribution Analysis:
The per-request analysis shows that most users get an excellent experience with fresh content, while only a small percentage of power users experience higher duplicate rates (5-10% of requests). This indicates the last_fallback system is working effectively.

SPECIAL NOTE:
- We have limited max exploratory content to 80K
- WHY: a user who is scrolling past 80K items in a day is probably not human, over-engineering for bots is waste of our resources.

Conclusion:
The zero-interaction fallback system successfully addresses feed exhaustion while maintaining excellent content discovery rates. The system now provides 2-3x more content than requested, ensuring abundant fresh recommendations for power users without compromising the experience for regular users.