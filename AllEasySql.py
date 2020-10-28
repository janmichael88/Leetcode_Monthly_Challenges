########################
#Bank Account Summary II
########################
# Write your MySQL query statement below
select u.name, SUM(t.amount) as balance
from Transactions t
join Users u
on u.account = t.account
group by u.name
having sum(t.amount) > 10000;

####################
# 1350. Students With Invalid Departments
####################
# Write your MySQL query statement below
select id,name
from Students
where department_id not in (Select id from Departments)

##########################
#1581. Customer Who Visited but Did Not Make Any Transactions
#########################
# Write your MySQL query statement below
select customer_id, count(*) as count_no_trans
from Visits
where visit_id not in (select visit_id from Transactions)
group by customer_id;

##########################
# 1571. Warehouse Manager
#########################
# Write your MySQL query statement below
select w.name as warehouse_name, sum(w.units*p.Width*p.Length*p.Height) as volume
from Warehouse w
inner join Products p
on w.product_id = p.product_id
group by w.name;

#######################################################
#1378. Replace Employee ID With The Unique Identifier
######################################################
# Write your MySQL query statement below
select Emp2.unique_id, Emp1.name
from Employees Emp1
left join EmployeeUNI Emp2
on Emp1.id = Emp2.id;

############################
#1301. Find the Team Size
############################
# Write your MySQL query statement below
select employee_id, COUNT(employee_id) OVER (PARTITION BY team_id) as team_size
from Employee;

#################################
#1527. Pateitns With a Condition
#################################
# Write your MySQL query statement below
select patient_id, patient_name, conditions
from Patients
WHERE conditions REGEXP "DIAB1";


########################################################
#1623 All Valid Triplets That Can Represent a Country
#########################################################
# Write your MySQL query statement below
select A.student_name as member_A, B.student_name as member_B, C.student_name as member_C
from SchoolA A, SchoolB B, SchoolC C
where A.student_name != B.student_name and A.student_name != C.student_name
and B.student_name != C.student_name and A.student_id != B.student_id and
A.student_id != C.student_id and B.student_id != C.student_id

#########################################
#1484. Group Sold Products By The Date
###########################################
# Write your MySQL query statement below
select sell_date, count(distinct(product)) as num_sold, group_concat(distinct(product)) as products
from activities
group by sell_date;

##################################
#1069. Product Sales Analysis II
###################################
# Write your MySQL query statement below
select product_id, sum(quantity) as total_quantity
from Sales
group by product_id;

###############################################
#1068. Product Sales Analysis I
##########################################
# Write your MySQL query statement below
select P.product_name, S.year, S.price
from Sales S
inner join Product P on S.product_id = P.product_id;


####################################
#1407. Top Travellers
####################################
# Write your MySQL query statement below
select U.name, Ifnull(SUM(R.distance),0) as travelled_distance
from Users U
left join Rides R on U.id = R.user_id
group by U.name
order by travelled_distance desc, U.name asc;


###################################
#1565. Unique Orders and Customers Per Month
###################################
# Write your MySQL query statement below
select Left(order_date,7) as month, count(distinct(order_id)) as order_count, count(distinct(customer_id)) as customer_count
from Orders
where invoice >20
group by month;

################################
#1251. Average Selling Price
################################
# Write your MySQL query statement below
select P.product_id, round(sum(P.price*U.units) / sum(U.units),2) as average_price
from Prices P
join UnitsSold U 
on P.product_id = U.product_id
where U.purchase_date between P.start_date and P.end_date
group by P.product_id;

######################################
#1179. Reformat Department Table
######################################
# Write your MySQL query statement below
#this is just pivoting a table
SELECT id,
    sum(case when month = 'Jan' then Revenue end) as Jan_Revenue,
    sum(case when month = 'Feb' then Revenue end) as Feb_Revenue,
    sum(case when month = 'Mar' then Revenue end) as Mar_Revenue,
    sum(case when month = 'Apr' then Revenue end) as Apr_Revenue,
    sum(case when month = 'May' then Revenue end) as May_Revenue,
    sum(case when month = 'Jun' then Revenue end) as Jun_Revenue,
    sum(case when month = 'Jul' then Revenue end) as Jul_Revenue,
    sum(case when month = 'Aug' then Revenue end) as Aug_Revenue,
    sum(case when month = 'Sep' then Revenue end) as Sep_Revenue,
    sum(case when month = 'Oct' then Revenue end) as Oct_Revenue,
    sum(case when month = 'Nov' then Revenue end) as Nov_Revenue,
    sum(case when month = 'Dec' then Revenue end) as Dec_Revenue
FROM Department
GROUP BY id

##################################
# 1173 Immediate Food Delivery I
##################################
# Write your MySQL query statement below
select round(100*sum(case when order_date = customer_pref_delivery_date then 1 else 0 end) / count(*),2) as immediate_percentage
from Delivery;


###########################
# 511. Game Play Analysis I
###########################
# Write your MySQL query statement below
select player_id, min(event_date) as first_login
from Activity
group by player_id;


########################################
# 613. Shortest Distance in a Line
########################################
# Write your MySQL query statement below
select min(dist) as shortest
from(select abs(x - lag(x)  over(order by x)) as dist
from point) as temp


########################
# 595. Big Countries
########################
# Write your MySQL query statement below
select name,area,population
from World
where area > 3000000 or population > 25000000

#################################
#1435. Create a Session Bar Chart
#################################
SELECT '[0-5>' AS bin, SUM(CASE WHEN duration >= 0 AND duration < 300 THEN 1 ELSE 0 END) AS total FROM sessions UNION
SELECT '[5-10>' AS bin, SUM(CASE WHEN duration >= 300 and duration < 600 THEN 1 ELSE 0 END) AS total FROM sessions UNION
SELECT '[10-15>' AS bin, SUM(CASE WHEN duration >= 600 and duration <= 900 THEN 1 ELSE 0 END) AS total FROM sessions UNION
SELECT '15 or more' AS bin, SUM(CASE WHEN duration >= 900 THEN 1 ELSE 0 END) as total FROM sessions

############################################
#1327. List the Products Ordered in a Period
############################################
# Write your MySQL query statement below
select P.product_name, sum(O.unit) as unit
from Products P
inner join Orders O
on P.product_id = O.product_id
where left(order_date, 7) = '2020-02'
group by P.product_name
having unit >= 100;